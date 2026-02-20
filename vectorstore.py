"""
Vector store: embeds document chunks and stores them in ChromaDB.
"""

import argparse
from pathlib import Path

DB_PATH = "output/vectorstore"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
COLLECTION_NAME = "documents"


def create_empty(db_path: str = DB_PATH):
    """Create an empty ChromaDB collection (no vectors).

    Used when there are no PDFs to process yet but the server needs to start.
    Documents can be added later via the /upload endpoint.
    """
    import chromadb

    Path(db_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except Exception:
        client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Empty collection '{COLLECTION_NAME}' created at '{db_path}'.")


def build(chunks: list[dict], db_path: str = DB_PATH):
    """Embed chunks and persist to ChromaDB.

    Args:
        chunks: list of dicts with keys: id, text, metadata
        db_path: directory where ChromaDB stores its files
    """
    import chromadb
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    print(f"Building vector store from {len(chunks)} chunks...")

    print(f"Loading embedding model {EMBEDDING_MODEL}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    batch_size = 32
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[i : i + batch_size]
        embeddings = embedder.encode(batch, normalize_embeddings=True).tolist()
        all_embeddings.extend(embeddings)

    collection.add(
        ids=ids,
        embeddings=all_embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Vector store built at '{db_path}' ({len(chunks)} vectors)")


def query(question, db_path=DB_PATH, n_results=3):
    """Return top-n most relevant chunks for the given question."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = embedder.encode(question, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "score": round(1 - dist, 4),  # cosine similarity
            "doc": meta["doc"],
            "page": meta["page"],
            "text": doc,
        })
    return hits


def main():
    parser = argparse.ArgumentParser(description="Vector store query tool")
    parser.add_argument("--query", type=str, required=True, metavar="QUESTION", help="Query the vector store")
    parser.add_argument("--db", default=DB_PATH, help="Path to vector store directory")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results (default: 3)")
    args = parser.parse_args()

    hits = query(args.query, db_path=args.db, n_results=args.top_k)
    for i, hit in enumerate(hits, 1):
        print(f"\n--- Result {i}  (score: {hit['score']}) ---")
        print(f"Doc: {hit['doc']}  |  Page: {hit['page']}")
        print(hit["text"][:600] + ("..." if len(hit["text"]) > 600 else ""))


if __name__ == "__main__":
    main()
