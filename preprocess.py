"""
Document preprocessing: extract text from PDFs using MiniCPM-o-4.5 (vision-only).

Phase 1 — extract:  renders each PDF page as image and runs it through the vision model
Phase 2 — chunk:    converts page-level extractions into chunks ready for vectorization
"""

import argparse
import io
import time
from pathlib import Path

import fitz
import yaml
from tqdm import tqdm

import vectorstore as vs


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_pdfs(docs_dir):
    docs = sorted(Path(docs_dir).glob("*.pdf"))
    if not docs:
        tqdm.write(f"No PDFs found in {docs_dir}/")
    return docs


VISION_PROMPT = (
    "Extract ALL content from this technical document page. "
    "For flowcharts and decision trees: describe the full logic flow "
    "as a structured list with conditions and outcomes. "
    "For tables: output as markdown tables. "
    "For regular text: preserve headings, lists, part numbers, "
    "measurements, and warnings. "
    "Output in markdown format."
)


def load_model():
    """Load MiniCPM-o-4.5 in vision-only mode."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "openbmb/MiniCPM-o-4_5"

    tqdm.write(f"Loading {model_name}...")
    t0 = time.time()
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=False,
        init_tts=False,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tqdm.write(f"Model loaded ({time.time() - t0:.1f}s)\n")
    return model, tokenizer


def extract(pdf_path, output_dir, config, page_num=None, model=None, tokenizer=None):
    """Extract all pages via vision model.

    Saves full document markdown to output/minicpm-o-4_5/{doc_name}.md.
    Returns a list of page dicts ready for chunk().
    """
    import torch
    from PIL import Image

    doc = fitz.open(pdf_path)
    doc_name = pdf_path.stem

    dpi = config.get("dpi", 200)
    pages = [page_num - 1] if page_num is not None else list(range(len(doc)))
    pages = [i for i in pages if 0 <= i < len(doc)]

    page_results = []
    with tqdm(pages, desc=doc_name, unit="page", leave=True) as pbar:
        for i in pbar:
            page = doc[i]
            pixmap = page.get_pixmap(dpi=dpi)
            image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")

            msgs = [{"role": "user", "content": [image, VISION_PROMPT]}]
            with torch.no_grad():
                text = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)

            page_results.append({
                "text": text,
                "page": i + 1,
                "doc": doc_name,
                "source": str(pdf_path),
            })
            pbar.set_postfix(chars=len(text))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    tqdm.write(f"Saved: {doc_name}\n")

    return page_results


def chunk(pages, min_chars=50, max_chars=8000):
    """Convert page-level extractions into chunks ready for vectorization.

    - Skips near-blank pages (< min_chars)
    - Truncates runaway pages (> max_chars) caused by model looping
    - Each chunk gets a stable id: {doc}_p{page:03d}
    """
    chunks = []
    for p in pages:
        text = p["text"].strip()
        if len(text) < min_chars:
            continue
        if len(text) > max_chars:
            text = text[:max_chars]
        chunks.append({
            "id": f"{p['doc']}_p{p['page']:03d}",
            "text": text,
            "metadata": {
                "doc": p["doc"],
                "page": p["page"],
                "source": p["source"],
            },
        })
    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from PDFs using MiniCPM-o-4.5 (vision-only)"
    )
    parser.add_argument(
        "--page",
        type=int,
        default=None,
        help="Extract a single page number (1-indexed, for testing)",
    )
    args = parser.parse_args()

    config = load_config()
    docs_dir = config.get("docs_dir", "docs")
    output_dir = config.get("output_dir", "output")
    pdfs = find_pdfs(docs_dir)

    if not pdfs:
        return

    model, tokenizer = load_model()

    all_chunks = []
    for pdf_path in tqdm(pdfs, desc="Documents", unit="doc"):
        pages = extract(
            pdf_path, output_dir, config,
            page_num=args.page, model=model, tokenizer=tokenizer,
        )
        all_chunks.extend(chunk(pages))

    tqdm.write(f"\n{len(all_chunks)} chunks ready — building vector store...")
    db_path = config.get("vectorstore_dir", vs.DB_PATH)
    vs.build(all_chunks, db_path=db_path)
    tqdm.write("Done.")


if __name__ == "__main__":
    main()
