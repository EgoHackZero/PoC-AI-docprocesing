
# PoC: Document Preprocessing — Parser Comparison

Compare MiniCPM-o-4.5 (vision-based) vs Docling for extracting text from technical PDF manuals, especially pages with flowcharts and decision trees.

## VRAM Requirements

| Parser | Min VRAM | Recommended | Notes |
|--------|----------|-------------|-------|
| `minicpm-v-hf` (MiniCPM-o-4.5, vision-only) | 16 GB | **24 GB** | Full bfloat16, flash_attn enabled. 16 GB works but may be slow on long documents due to tight memory. 24 GB gives smooth throughput. |
| `internvl` (InternVL2-2B) | 6 GB | 8 GB | Lightweight but lower quality on complex flowcharts. |
| `docling` | — | 4 GB | GPU accelerates layout analysis and OCR models. Falls back to CPU automatically if no GPU. |

> For MiniCPM-o-4.5 (vision-only) with flash_attn on 16 GB VRAM: expect ~10–20s/page.
> On 24 GB VRAM (e.g. RTX 3090 / A10G): expect ~5–10s/page.

## Usage

Place PDF files in `docs/`.

```bash
# Recommended: MiniCPM-o-4.5 (vision-only) via HuggingFace (best quality on flowcharts/diagrams)
python preprocess.py --parser minicpm-v-hf

# Test a single page first
python preprocess.py --parser minicpm-v-hf --page 17

# Alternative: Docling (GPU-accelerated, loses flowchart structure)
python preprocess.py --parser docling

# Alternative: InternVL2-2B (low VRAM, lower quality)
python preprocess.py --parser internvl
```

Output is saved to `output/{parser}/{document_name}/page_NNN.md`.

## Configuration

Edit `config.yaml`:

```yaml
docs_dir: "docs"
output_dir: "output"
dpi: 200
```

## Parser Comparison (PoC findings)

| Parser | Flowcharts | Tables | Plain text | Speed |
|--------|-----------|--------|------------|-------|
| MiniCPM-o-4.5 HF (vision-only) | ✅ Full logic extracted | ✅ Markdown | ✅ | ~10s/page (24GB GPU) |
| InternVL2-2B HF | ⚠️ Correct content, some hallucination | ✅ | ✅ | ~60s/page (6GB GPU) |
| Docling | ❌ Replaced with `<!-- image -->` | ✅ | ✅ | ~5 min/doc (CPU) / ~30–60s/doc (GPU) |
