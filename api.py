"""
FastAPI application for the document Q&A pipeline.
"""

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

import vectorstore as vs

load_dotenv()

DB_PATH = os.getenv("VECTORSTORE_PATH", vs.DB_PATH)
DOCS_DIR = "docs"

_upload_jobs: dict[str, dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(DOCS_DIR).mkdir(exist_ok=True)
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set — /query endpoint will fail.")
    if not Path(DB_PATH).exists():
        print("WARNING: Vector store not found. Run preprocess.py to build it.")
    yield


app = FastAPI(
    title="Document Q&A API",
    description="RAG pipeline over technical service manuals using CrewAI + GPT-4o-mini",
    version="1.0.0",
    lifespan=lifespan,
)

class QueryRequest(BaseModel):
    question: str
    db_path: str = DB_PATH

class QueryResponse(BaseModel):
    answer: str

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    message: str

class UploadStatusResponse(BaseModel):
    job_id: str
    filename: str
    status: str   # queued | processing | done | error
    detail: str


@app.get("/health")
def health():
    active_jobs = [j for j in _upload_jobs.values() if j["status"] in ("queued", "processing")]
    return {
        "status": "ok",
        "vectorstore_ready": Path(DB_PATH).exists(),
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "active_upload_jobs": len(active_jobs),
    }


@app.get("/documents")
def list_documents():
    if not Path(DB_PATH).exists():
        raise HTTPException(
            status_code=404,
            detail="Vector store not found. Run preprocess.py first.",
        )
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(vs.COLLECTION_NAME)
    results = collection.get(include=["metadatas"])

    doc_counts: dict[str, int] = {}
    for meta in results["metadatas"]:
        doc = meta["doc"]
        doc_counts[doc] = doc_counts.get(doc, 0) + 1

    return {
        "total_chunks": len(results["metadatas"]),
        "documents": [
            {"name": name, "chunks": count}
            for name, count in sorted(doc_counts.items())
        ],
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest = Path(DOCS_DIR) / file.filename
    content = await file.read()
    dest.write_bytes(content)

    job_id = str(uuid.uuid4())
    _upload_jobs[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "detail": "Saved to docs/, waiting for processing to start.",
    }

    background_tasks.add_task(_process_uploaded_file, job_id, dest)

    return UploadResponse(
        job_id=job_id,
        filename=file.filename,
        status="queued",
        message=f"File saved. Processing started in background. Poll GET /upload/status/{job_id}",
    )


@app.get("/upload/status/{job_id}", response_model=UploadStatusResponse)
def upload_status(job_id: str):
    if job_id not in _upload_jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = _upload_jobs[job_id]
    return UploadStatusResponse(
        job_id=job_id,
        filename=job["filename"],
        status=job["status"],
        detail=job["detail"],
    )


def _process_uploaded_file(job_id: str, pdf_path: Path):
    """Background task: extract → chunk → upsert into existing vector store."""
    import preprocess

    _upload_jobs[job_id]["status"] = "processing"
    _upload_jobs[job_id]["detail"] = "Loading vision model and extracting pages..."

    try:
        config = preprocess.load_config()
        output_dir = config.get("output_dir", "output")

        model, tokenizer = preprocess.load_model()
        pages = preprocess.extract(
            pdf_path, output_dir, config,
            model=model, tokenizer=tokenizer,
        )
        chunks = preprocess.chunk(pages)

        _upload_jobs[job_id]["detail"] = f"Extracted {len(pages)} pages, embedding {len(chunks)} chunks..."

        # Upsert into existing collection (add new, overwrite if same id)
        import chromadb
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(vs.EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=DB_PATH)

        try:
            collection = client.get_collection(vs.COLLECTION_NAME)
        except Exception:
            collection = client.create_collection(
                name=vs.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        texts = [c["text"] for c in chunks]
        ids = [c["id"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()

        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        _upload_jobs[job_id]["status"] = "done"
        _upload_jobs[job_id]["detail"] = (
            f"Done. {len(chunks)} chunks added/updated in vector store."
        )

    except Exception as e:
        _upload_jobs[job_id]["status"] = "error"
        _upload_jobs[job_id]["detail"] = str(e)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not set.")
    if not Path(req.db_path).exists():
        raise HTTPException(
            status_code=503,
            detail="Vector store not found. Run preprocess.py first.",
        )

    from agents import run as crew_run

    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        None, lambda: crew_run(req.question, db_path=req.db_path)
    )
    return QueryResponse(answer=answer)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
