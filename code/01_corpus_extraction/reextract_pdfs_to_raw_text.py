#!/usr/bin/env python3
"""
Re-run PDF → UTF-8 text for PDFs already under data/pdfs/ (no download).

Examples
--------
    # Improved digital extraction only (default)
    python3 tools/reextract_pdfs_to_raw_text.py --output-dir .

    # Hybrid OCR for scan-heavy PDFs (needs: brew install tesseract)
    python3 tools/reextract_pdfs_to_raw_text.py --output-dir . --ocr --workers 2

    # Single file
    python3 tools/reextract_pdfs_to_raw_text.py --output-dir . --only K120161
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pdf_text_extract import ocr_dependencies_ok, pdf_to_text_with_options


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-extract raw_text from local PDFs.")
    p.add_argument(
        "--output-dir",
        default=".",
        type=Path,
        help="Project root containing data/pdfs and data/raw_text.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers (use 1–2 with --ocr).",
    )
    p.add_argument(
        "--only",
        default="",
        help="Comma-separated stems (e.g. K120161,K171056); empty = all PDFs.",
    )
    p.add_argument(
        "--ocr",
        action="store_true",
        help="Enable Tesseract (hybrid unless --ocr-mode full).",
    )
    p.add_argument(
        "--ocr-mode",
        choices=("hybrid", "full", "off"),
        default="hybrid",
    )
    return p.parse_args()


def _stem_filter(raw: str) -> set[str] | None:
    if not raw.strip():
        return None
    out = {x.strip().upper() for x in raw.split(",") if x.strip()}
    return out or None


def _one_pdf(pdf_path: Path, text_dir: Path, use_ocr: bool, ocr_mode: str) -> tuple[str, str]:
    stem = pdf_path.stem.upper()
    try:
        data = pdf_path.read_bytes()
        if not data.startswith(b"%PDF-"):
            return stem, "skip: not a PDF"
        text = pdf_to_text_with_options(data, use_ocr=use_ocr, ocr_mode=ocr_mode)
        out = text_dir / f"{pdf_path.stem}.txt"
        out.write_text(text, encoding="utf-8")
        return stem, "ok"
    except Exception as exc:
        return stem, f"error: {exc}"


def main() -> None:
    args = parse_args()
    base = args.output_dir.resolve()
    pdf_dir = base / "data" / "pdfs"
    text_dir = base / "data" / "raw_text"
    if not pdf_dir.is_dir():
        raise SystemExit(f"Missing directory: {pdf_dir}")

    text_dir.mkdir(parents=True, exist_ok=True)
    use_ocr = bool(args.ocr) and args.ocr_mode != "off"
    if use_ocr and not ocr_dependencies_ok():
        raise SystemExit(
            "OCR requested but Tesseract or Python bindings missing. "
            "Install: brew install tesseract && pip install pytesseract Pillow"
        )

    only = _stem_filter(args.only)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if only is not None:
        pdfs = [p for p in pdfs if p.stem.upper() in only]

    if not pdfs:
        print("No PDFs matched.")
        return

    print(f"Re-extracting {len(pdfs)} PDF(s); digital+heuristic; OCR={use_ocr} mode={args.ocr_mode}")

    if args.workers <= 1:
        for pdf in pdfs:
            stem, status = _one_pdf(pdf, text_dir, use_ocr, args.ocr_mode)
            print(f"{stem}: {status}")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_one_pdf, pdf, text_dir, use_ocr, args.ocr_mode): pdf for pdf in pdfs
        }
        for fut in as_completed(futs):
            stem, status = fut.result()
            print(f"{stem}: {status}")


if __name__ == "__main__":
    main()
