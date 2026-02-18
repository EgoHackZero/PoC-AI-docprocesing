"""
Document preprocessing: extract text from PDFs using MiniCPM-V or Docling.

Usage:
    python preprocess.py --parser minicpm-v              # All PDFs, MiniCPM-V via Ollama
    python preprocess.py --parser minicpm-v-hf           # All PDFs, MiniCPM-V via HuggingFace
    python preprocess.py --parser docling                # All PDFs, Docling
    python preprocess.py --parser internvl               # All PDFs, InternVL2-1B via HuggingFace
    python preprocess.py --parser minicpm-v-hf --page 17 # Single page (for testing)
"""

import argparse
import io
import time
from pathlib import Path

import fitz  # PyMuPDF
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_pdfs(docs_dir):
    docs = sorted(Path(docs_dir).glob("*.pdf"))
    if not docs:
        print(f"No PDFs found in {docs_dir}/")
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


def extract_minicpmv(pdf_path, output_dir, config, page_num=None):
    """Extract text from PDF pages using MiniCPM-V via Ollama."""
    import ollama

    doc = fitz.open(pdf_path)
    doc_name = pdf_path.stem
    out_dir = Path(output_dir) / "minicpm-v" / doc_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dpi = config.get("dpi", 200)
    model = config.get("vision_model", "minicpm-v")

    if page_num is not None:
        pages = [page_num - 1]
    else:
        pages = range(len(doc))

    total_start = time.time()

    for i in pages:
        if i < 0 or i >= len(doc):
            print(f"Page {i + 1} out of range (document has {len(doc)} pages)")
            continue

        page_start = time.time()
        page = doc[i]
        pixmap = page.get_pixmap(dpi=dpi)
        image_bytes = pixmap.tobytes("png")

        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": VISION_PROMPT,
                    "images": [image_bytes],
                }
            ],
        )

        text = response["message"]["content"]
        out_file = out_dir / f"page_{i + 1:03d}.md"
        out_file.write_text(text, encoding="utf-8")

        elapsed = time.time() - page_start
        print(f"  [page {i + 1}/{len(doc)}] extracted ({len(text)} chars, {elapsed:.1f}s)")

    total_elapsed = time.time() - total_start
    print(f"  Done: {doc_name} ({total_elapsed:.1f}s total)")


def load_minicpmv_hf():
    """Load MiniCPM-o-4.5 model in vision-only mode and tokenizer once."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "openbmb/MiniCPM-o-4_5"

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    print(f"Loading model {model_name} (attn: {attn_impl})...")
    load_start = time.time()
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=False,
        init_tts=False,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"Model loaded ({time.time() - load_start:.1f}s)")
    return model, tokenizer


def extract_minicpmv_hf(pdf_path, output_dir, config, page_num=None, model=None, tokenizer=None):
    """Extract text from PDF pages using MiniCPM-V-2.6 via HuggingFace transformers."""
    import torch
    from PIL import Image

    doc = fitz.open(pdf_path)
    doc_name = pdf_path.stem
    out_dir = Path(output_dir) / "minicpm-v-hf" / doc_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dpi = config.get("dpi", 200)

    if page_num is not None:
        pages = [page_num - 1]
    else:
        pages = range(len(doc))

    total_start = time.time()

    for i in pages:
        if i < 0 or i >= len(doc):
            print(f"Page {i + 1} out of range (document has {len(doc)} pages)")
            continue

        page_start = time.time()
        page = doc[i]
        pixmap = page.get_pixmap(dpi=dpi)
        png_bytes = pixmap.tobytes("png")
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")

        msgs = [{"role": "user", "content": [image, VISION_PROMPT]}]

        with torch.no_grad():
            text = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)

        out_file = out_dir / f"page_{i + 1:03d}.md"
        out_file.write_text(text, encoding="utf-8")

        elapsed = time.time() - page_start
        print(f"  [page {i + 1}/{len(doc)}] extracted ({len(text)} chars, {elapsed:.1f}s)")

        # Free GPU memory between pages
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    print(f"  Done: {doc_name} ({total_elapsed:.1f}s total)")


def load_internvl():
    """Load InternVL2-2B model and tokenizer once."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "OpenGVLab/InternVL2-2B"
    print(f"Loading model {model_name}...")
    load_start = time.time()

    try:
        import flash_attn  # noqa: F401
        use_flash = True
    except ImportError:
        use_flash = False

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )

    print(f"Model loaded ({time.time() - load_start:.1f}s)")
    return model, tokenizer


def _internvl_build_transform(input_size):
    """Build image transform pipeline for InternVL2."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448):
    """Split image into tiles for InternVL2 dynamic high-resolution processing."""
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    # Find best tile grid that matches the aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = min(
        target_ratios,
        key=lambda r: abs(aspect_ratio - r[0] / r[1]),
    )
    target_w = image_size * best_ratio[0]
    target_h = image_size * best_ratio[1]

    resized = image.resize((target_w, target_h))
    num_cols, num_rows = best_ratio
    tiles = []
    for row in range(num_rows):
        for col in range(num_cols):
            box = (
                col * image_size,
                row * image_size,
                (col + 1) * image_size,
                (row + 1) * image_size,
            )
            tiles.append(resized.crop(box))

    # Add a thumbnail of the full image
    thumbnail = image.resize((image_size, image_size))
    tiles.append(thumbnail)
    return tiles


def extract_internvl(pdf_path, output_dir, config, page_num=None, model=None, tokenizer=None):
    """Extract text from PDF pages using InternVL2-2B via HuggingFace."""
    import torch
    from PIL import Image

    doc = fitz.open(pdf_path)
    doc_name = pdf_path.stem
    out_dir = Path(output_dir) / "internvl" / doc_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dpi = config.get("dpi", 200)
    image_size = 448
    transform = _internvl_build_transform(image_size)

    generation_config = dict(max_new_tokens=1024, do_sample=False)

    if page_num is not None:
        pages = [page_num - 1]
    else:
        pages = range(len(doc))

    total_start = time.time()

    for i in pages:
        if i < 0 or i >= len(doc):
            print(f"Page {i + 1} out of range (document has {len(doc)} pages)")
            continue

        page_start = time.time()
        page = doc[i]
        pixmap = page.get_pixmap(dpi=dpi)
        png_bytes = pixmap.tobytes("png")
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")

        tiles = _internvl_dynamic_preprocess(image, image_size=image_size)
        pixel_values = torch.stack([transform(t) for t in tiles]).to(
            dtype=torch.bfloat16, device="cuda"
        )

        question = f"<image>\n{VISION_PROMPT}"

        with torch.no_grad():
            text = model.chat(tokenizer, pixel_values, question, generation_config)

        out_file = out_dir / f"page_{i + 1:03d}.md"
        out_file.write_text(text, encoding="utf-8")

        elapsed = time.time() - page_start
        print(f"  [page {i + 1}/{len(doc)}] extracted ({len(text)} chars, {elapsed:.1f}s)")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    print(f"  Done: {doc_name} ({total_elapsed:.1f}s total)")


def extract_docling(pdf_path, output_dir, page_num=None):
    """Extract text from PDF using Docling's DocumentConverter."""
    from docling.document_converter import DocumentConverter

    doc_name = pdf_path.stem
    out_dir = Path(output_dir) / "docling" / doc_name
    out_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    markdown = result.document.export_to_markdown()

    if page_num is not None:
        print(f"  Note: Docling processes the entire document. --page filter saves full output.")

    out_file = out_dir / "full_document.md"
    out_file.write_text(markdown, encoding="utf-8")

    total_elapsed = time.time() - total_start
    print(f"  Done: {doc_name} ({len(markdown)} chars, {total_elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDFs")
    parser.add_argument(
        "--parser",
        required=True,
        choices=["minicpm-v", "minicpm-v-hf", "docling", "internvl"],
        help="Parser to use",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=None,
        help="Extract a single page number (1-indexed)",
    )
    args = parser.parse_args()

    config = load_config()
    docs_dir = config.get("docs_dir", "docs")
    output_dir = config.get("output_dir", "output")
    pdfs = find_pdfs(docs_dir)

    if not pdfs:
        return

    print(f"Parser: {args.parser}")
    print(f"PDFs found: {len(pdfs)}")
    if args.page:
        print(f"Page filter: {args.page}")
    print()

    # Load HF model once if needed
    hf_model, hf_tokenizer = None, None
    if args.parser == "minicpm-v-hf":
        hf_model, hf_tokenizer = load_minicpmv_hf()
    elif args.parser == "internvl":
        hf_model, hf_tokenizer = load_internvl()

    for pdf_path in pdfs:
        print(f"Processing: {pdf_path.name}")
        if args.parser == "minicpm-v":
            extract_minicpmv(pdf_path, output_dir, config, page_num=args.page)
        elif args.parser == "minicpm-v-hf":
            extract_minicpmv_hf(
                pdf_path, output_dir, config,
                page_num=args.page, model=hf_model, tokenizer=hf_tokenizer,
            )
        elif args.parser == "docling":
            extract_docling(pdf_path, output_dir, page_num=args.page)
        elif args.parser == "internvl":
            extract_internvl(
                pdf_path, output_dir, config,
                page_num=args.page, model=hf_model, tokenizer=hf_tokenizer,
            )
        print()


if __name__ == "__main__":
    main()
