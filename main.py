"""
Unified entry point for the document Q&A pipeline.

Database persistence:
    ChromaDB stores everything in output/vectorstore/ on disk.
    After a restart, run `python main.py serve` — data is already there,
    no need to reprocess. Use --force-preprocess to rebuild from scratch.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def run_preprocess(page=None):
    import preprocess
    import vectorstore as vs
    from tqdm import tqdm

    config = preprocess.load_config()
    docs_dir = config.get("docs_dir", "docs")
    output_dir = config.get("output_dir", "output")

    pdfs = preprocess.find_pdfs(docs_dir)
    if not pdfs:
        return False

    model, tokenizer = preprocess.load_model()

    all_chunks = []
    for pdf_path in tqdm(pdfs, desc="Documents", unit="doc"):
        pages = preprocess.extract(
            pdf_path, output_dir, config,
            page_num=page, model=model, tokenizer=tokenizer,
        )
        all_chunks.extend(preprocess.chunk(pages))

    tqdm.write(f"\n{len(all_chunks)} chunks ready — building vector store...")
    db_path = config.get("vectorstore_dir", vs.DB_PATH)
    vs.build(all_chunks, db_path=db_path)
    tqdm.write("Preprocessing complete.\n")
    return True


def run_serve(port=8000):
    import uvicorn
    print(f"Starting API server → http://0.0.0.0:{port}")
    print(f"Swagger UI         → http://0.0.0.0:{port}/docs\n")
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)


def run_query(question):
    import vectorstore as vs

    if not Path(vs.DB_PATH).exists():
        print("ERROR: Vector store not found. Run `python main.py` first.")
        sys.exit(1)
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Add it to .env")
        sys.exit(1)

    from agents import run as crew_run
    print(f"Question: {question}\n")
    print("Answer:")
    print(crew_run(question))


def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Document Q&A pipeline — extract, embed, and query service manuals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Default behaviour (no subcommand) flags
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Re-run preprocessing even if the vector store already exists",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the API server when running in full mode (default: 8000)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # preprocess
    p_pre = subparsers.add_parser("preprocess", help="Extract PDFs and build vector store")
    p_pre.add_argument("--page", type=int, default=None,
                       help="Process a single page only (1-indexed, for testing)")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start the FastAPI server")
    p_serve.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")

    # query
    p_query = subparsers.add_parser("query", help="Ask a question from the CLI")
    p_query.add_argument("question", type=str)

    args = parser.parse_args()

    if args.command == "preprocess":
        run_preprocess(page=args.page)

    elif args.command == "serve":
        import vectorstore as vs
        if not Path(vs.DB_PATH).exists():
            print("ERROR: Vector store not found. Run `python main.py preprocess` first.")
            sys.exit(1)
        run_serve(port=args.port)

    elif args.command == "query":
        run_query(args.question)

    else:
        import preprocess
        import vectorstore as vs

        db_exists = Path(vs.DB_PATH).exists()

        if db_exists and not args.force_preprocess:
            # DB already on disk — start immediately, no preprocessing
            print("Vector store found — starting server immediately.")
            print("Use --force-preprocess to rebuild from scratch.\n")

        else:
            if args.force_preprocess:
                print("--force-preprocess set — rebuilding vector store.\n")

            config = preprocess.load_config()
            pdfs = preprocess.find_pdfs(config.get("docs_dir", "docs"))

            if pdfs:
                # Files available — preprocess then start
                print(f"Found {len(pdfs)} PDF(s) — running preprocessing...\n")
                run_preprocess()
            else:
                # No files yet — create empty DB and wait for uploads
                print("No PDFs found in docs/ — creating empty vector store.")
                print("Add files via POST /upload once the server is running.\n")
                vs.create_empty()

        run_serve(port=args.port)


if __name__ == "__main__":
    main()