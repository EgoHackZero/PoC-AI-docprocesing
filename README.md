# PoC: AI Document Processing Pipeline

End-to-end RAG pipeline for technical service manuals. Extracts content from PDFs containing flowcharts and decision trees using a vision LLM, stores it in a vector database, and answers natural language questions via a CrewAI agent crew backed by GPT-4o-mini.

**Use case:** A technician asks *"What should I check when E12 alarm appears?"* and gets a structured, step-by-step answer sourced directly from the service manual.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                          │
│                                                                 │
│  PDF pages                                                      │
│     │                                                           │
│     ▼                                                           │
│  [Stage 1] MiniCPM-o-4.5 (vision-only)                         │
│     │  Renders each page as PNG → extracts markdown             │
│     ▼                                                           │
│  [Stage 2] Chunker                                              │
│     │  Page = chunk | skip blanks | truncate loops              │
│     ▼                                                           │
│  [Stage 3] BAAI/bge-large-en-v1.5 + ChromaDB                   │
│            Embeds chunks → persists to vector store             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE  (CrewAI)                     │
│                                                                 │
│  User question (may contain typos)                              │
│     │                                                           │
│     ▼                                                           │
│  [Agent 1] Query Rewriter  (GPT-4o-mini)                        │
│     │  Fixes spelling  →  generates 3-5 query variants          │
│     ▼                                                           │
│  [Agent 2] Retriever  (GPT-4o-mini + SearchManualTool)          │
│     │  Runs each query variant against ChromaDB                 │
│     │  Deduplicates results by (doc, page)                      │
│     ▼                                                           │
│  [Agent 3] Evaluator  (GPT-4o-mini)                             │
│     │  Verdict: SUFFICIENT or INSUFFICIENT                      │
│     ▼                                                           │
│  [Agent 4] Specialist  (GPT-4o-mini)                            │
│     │  SUFFICIENT  → structured answer with source citations    │
│     │  INSUFFICIENT → "no information about this in database"   │
│     ▼                                                           │
│  FastAPI  →  POST /query  →  final answer                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project structure

```
├── main.py          — unified CLI entry point
├── preprocess.py    — Stage 1+2: PDF extraction + chunking
├── vectorstore.py   — Stage 3: embedding + ChromaDB
├── agents.py        — Stage 4: CrewAI agents (retriever + specialist)
├── api.py           — FastAPI server
├── config.yaml      — paths and DPI settings
├── requirements.txt
├── .env.example
└── docs/            — place PDF files here (not tracked in git)
```

---

## VRAM requirements

| Stage | Component | Min VRAM | Recommended |
|-------|-----------|----------|-------------|
| Extraction | MiniCPM-o-4.5 (vision-only, bfloat16) | 16 GB | **24–40 GB** |
| Embedding | BAAI/bge-large-en-v1.5 | 2 GB | 4 GB |
| Inference | GPT-4o-mini (OpenAI API) | — | — |

> Extraction was benchmarked on **NVIDIA A100-SXM4-40GB**. Embedding and inference do not require a large GPU.

---

## Benchmark results (production run on A100 40GB, CUDA 12.9)

| Document | Pages | Total time | Avg per page |
|----------|-------|------------|--------------|
| LAD-Front-Loading-Service-Manual-L11 | 71 | 14 min 59s | ~12.7s |
| technical-manual-w11663204-revb | 66 | 29 min 13s | ~26.6s |

- Model load time: **8.2s**
- Both documents: **~44 min total**
- Speed varies by content: plain text ~4s, complex flowcharts/tables 20–70s
- ~3 pages across both docs triggered model looping (truncated at 8 000 chars by the chunker)

---

## Parser comparison (PoC findings)

| Parser | Flowcharts | Tables | Plain text | Speed |
|--------|-----------|--------|------------|-------|
| **MiniCPM-o-4.5** (selected) | ✅ Full logic, correct Yes/No branches | ✅ Markdown | ✅ | ~13s/page (A100) |
| MiniCPM-V-2.6-int4 | ⚠️ Correct text nodes, wrong nesting | ✅ | ✅ | ~192s/page (RTX 2060) |
| InternVL2-1B | ⚠️ Correct content, output duplication | ✅ | ✅ | ~66s/page (RTX 2060) |
| Docling | ❌ Flowcharts → `<!-- image -->` | ✅ | ⚠️ Font encoding issues on some PDFs | ~5 min/doc (CPU) |
| minicpm-v (Ollama Q4_0) | ❌ Garbage ASCII output | ❌ | ❌ | — |

**Why MiniCPM-o-4.5:** Only model that correctly extracted the full E12 decision tree with proper Yes/No branching — verified against the original diagram. Loaded in vision-only mode (`init_audio=False`, `init_tts=False`) to save VRAM.

---

## Installation

### 1. Prerequisites

```bash
nvidia-smi                          # verify CUDA driver
```

### 2. Clone and copy PDFs

```bash
git clone <repo-url>
cd <repo-name>

# PDFs are not tracked in git — copy them to docs/
```

### 3. Create environment

```bash
conda create -n docproc python=3.10 -y
conda activate docproc
```

### 4. Install PyTorch with CUDA

```bash
pip install "torch>=2.3.0,<=2.8.0" "torchaudio<=2.8.0" \
    --index-url https://download.pytorch.org/whl/cu126
```

### 5. Install project dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure environment

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```

---

## Usage

### Just run this — it does everything

```bash
python main.py
```

Startup logic:

| Situation | What happens |
|-----------|-------------|
| DB exists | Start server immediately — no preprocessing |
| DB missing, PDFs in `docs/` | Preprocess all PDFs → build DB → start server |
| DB missing, no PDFs | Create empty DB → start server → wait for uploads via `POST /upload` |

### After a restart — no reprocessing needed

```bash
python main.py          # skips preprocessing, starts server immediately
```

The vector store is saved to disk (`output/vectorstore/`). It persists between restarts.

### Force reprocess (e.g. after adding new PDFs to docs/)

```bash
python main.py --force-preprocess
```

### Custom port

```bash
python main.py --port 9000
```

### Individual commands

```bash
# Only preprocess (no server)
python main.py preprocess

# Test on a single page before full run
python main.py preprocess --page 17

# Only start server (DB must already exist)
python main.py serve
python main.py serve --port 9000

# Query from CLI without starting the server
python main.py query "What should I check when E12 alarm appears?"
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service status, vectorstore ready, active upload jobs |
| `GET` | `/documents` | Lists indexed documents and chunk counts |
| `POST` | `/upload` | Upload a new PDF — saves and processes it into the vector store |
| `GET` | `/upload/status/{job_id}` | Poll processing status of an uploaded file |
| `POST` | `/query` | Ask a question, returns answer from the crew |

### Upload a new document

Uploading saves the PDF, then runs the full extraction + embedding pipeline in the background. The vector store is updated with the new document without rebuilding from scratch (upsert).

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/new-manual.pdf"
```

```json
{
  "job_id": "a3f2c1d0-...",
  "filename": "new-manual.pdf",
  "status": "queued",
  "message": "File saved. Processing started in background. Poll GET /upload/status/a3f2c1d0-..."
}
```

**Poll for status:**

```bash
curl http://localhost:8000/upload/status/a3f2c1d0-...
```

```json
{
  "job_id": "a3f2c1d0-...",
  "filename": "new-manual.pdf",
  "status": "done",
  "detail": "Done. 58 chunks added/updated in vector store."
}
```

Status values: `queued` → `processing` → `done` / `error`

> Note: processing time depends on document size (~13–27s/page on A100). A 70-page manual takes ~15–30 min.

### Query example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "E12 alarm water level too high, what to check?"}'
```

```json
{
  "answer": "According to the LAD service manual (page 17), when E12 alarm appears:\n1. Restart the machine to see if it still alarms — if No, the machine is ok.\n2. Check whether water is entering without power — if Yes, replace the water valve.\n3. Check whether the air pipe or connection is leaking — if No, replace it.\n..."
}
```

---

## Configuration

`config.yaml`:

```yaml
docs_dir: "docs"
output_dir: "output"
vectorstore_dir: "output/vectorstore"
dpi: 200
```
