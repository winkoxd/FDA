"""
High-quality English PDF → plain text for FDA 510(k) summaries.

Strategy
--------
1. **Digital text (primary):** PyMuPDF with reading-order sort + dehyphenation, then
   light normalization (hyphenated line breaks, whitespace, bullets).
2. **Digital text (fallback):** pdfminer.six with layout parameters tuned for
   single-column prose.
3. **Pick the stronger digital layer** by English word-ish score (avoids choosing
   garbage font maps).
4. **Scanned / print PDFs (optional):** If a page has very little extractable text,
   render that page and run **Tesseract** (English). Requires system `tesseract`
   (e.g. ``brew install tesseract``) and optional deps ``pytesseract``, ``Pillow``.

OCR is **off by default** in the crawler to keep bulk downloads fast; enable with
``use_ocr=True`` or use ``tools/reextract_pdfs_to_raw_text.py --ocr``.
"""

from __future__ import annotations

import io
import re
import threading

_OCR_LOCK = threading.Lock()


def _english_word_score(text: str) -> int:
    """Count plausible English tokens (length ≥ 2); robust to noisy font maps."""
    return len(re.findall(r"\b[A-Za-z][A-Za-z']+\b", text))


def _postcorrect_fda_scan_artifacts(text: str) -> str:
    """
    Conservative fixes for recurring glyph / ToUnicode issues in older 510(k) PDFs.
    Does not rewrite clinical content—only obvious template strings.
    """
    t = text
    t = re.sub(r"\b21\s+CIFR\b", "21 CFR", t, flags=re.I)
    t = re.sub(r"(\bRegulatory\s+)Glass(\s*:)", r"\1Class\2", t, flags=re.I)
    # Glyph substitution: straight double-quote used as a bullet
    t = re.sub(r"(?m)^(\s*)\"\s+Sends\b", r"\1* Sends", t)
    t = re.sub(r"(?m)^(\s*)\"\s+The software\b", r"\1* The software", t)
    t = re.sub(r"\bJ-STID\s+016\b", "J-STD 016", t, flags=re.I)
    return t


def _normalize_common_ocr_noise(text: str) -> str:
    """Light cleanup after extraction (does not fix all OCR substitutions)."""
    # Ligatures / odd spaces
    t = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    t = t.replace("\u00a0", " ").replace("\u200b", "")
    # Normalise line endings
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # De-hyphenate: word-\nword (optional spaces) → wordword (common in PDF line breaks)
    t = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", t)
    # Collapse excessive blank lines (preserve paragraph breaks)
    t = re.sub(r"\n{4,}", "\n\n\n", t)
    # Strip trailing spaces on each line
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    return _postcorrect_fda_scan_artifacts(t).strip()


def extract_text_pymupdf(pdf_data: bytes) -> str:
    import fitz  # PyMuPDF

    parts: list[str] = []
    flags = fitz.TEXT_DEHYPHENATE
    with fitz.open(stream=pdf_data, filetype="pdf") as doc:
        for page in doc:
            try:
                raw = page.get_text("text", sort=True, flags=flags)
            except TypeError:
                raw = page.get_text("text", sort=True)
            parts.append(raw or "")
    return _normalize_common_ocr_noise("\n\n".join(parts))


def extract_text_pdfminer(pdf_data: bytes) -> str:
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams

    laparams = LAParams(
        line_margin=0.35,
        word_margin=0.15,
        char_margin=2.5,
        boxes_flow=0.5,
        all_texts=True,
    )
    buf = io.BytesIO(pdf_data)
    raw = extract_text(buf, laparams=laparams) or ""
    return _normalize_common_ocr_noise(raw)


def _best_digital_layer(pdf_data: bytes) -> tuple[str, str]:
    """Return (chosen_text, method_name)."""
    t_fitz = ""
    t_miner = ""
    try:
        t_fitz = extract_text_pymupdf(pdf_data)
    except Exception:
        t_fitz = ""
    try:
        t_miner = extract_text_pdfminer(pdf_data)
    except Exception:
        t_miner = ""

    s_f, s_m = _english_word_score(t_fitz), _english_word_score(t_miner)
    # Prefer PyMuPDF when scores are close (usually better glyph paths for FDA PDFs)
    if s_f >= max(s_m * 0.88, s_m - 15):
        return (t_fitz if t_fitz.strip() else t_miner, "pymupdf" if t_fitz.strip() else "pdfminer")
    return (t_miner if t_miner.strip() else t_fitz, "pdfminer" if t_miner.strip() else "pymupdf")


def _ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        return False
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_page_image(page: object, zoom: float = 2.25) -> str:
    """OCR a single PyMuPDF page object."""
    import fitz
    from PIL import Image
    import pytesseract

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    # OEM 3 = default LSTM; PSM 6 = single uniform block (typical 510k summary)
    with _OCR_LOCK:
        return pytesseract.image_to_string(img, lang="eng", config="--oem 3 --psm 6")


def extract_text_hybrid_per_page(
    pdf_data: bytes,
    *,
    min_words_digital_page: int = 28,
    zoom: float = 2.25,
) -> str:
    """
    Per-page: use digital text if word-rich; otherwise OCR that page.
    """
    import fitz

    if not _ocr_available():
        text, _ = _best_digital_layer(pdf_data)
        return text

    parts: list[str] = []
    flags = fitz.TEXT_DEHYPHENATE
    with fitz.open(stream=pdf_data, filetype="pdf") as doc:
        for page in doc:
            try:
                digital = page.get_text("text", sort=True, flags=flags)
            except TypeError:
                digital = page.get_text("text", sort=True)
            digital = digital or ""
            if _english_word_score(digital) >= min_words_digital_page:
                parts.append(_normalize_common_ocr_noise(digital))
            else:
                try:
                    ocr_t = _ocr_page_image(page, zoom=zoom)
                    ocr_t = _normalize_common_ocr_noise(ocr_t)
                    # If OCR is still empty, keep digital
                    if _english_word_score(ocr_t) > _english_word_score(digital):
                        parts.append(ocr_t)
                    else:
                        parts.append(_normalize_common_ocr_noise(digital))
                except Exception:
                    parts.append(_normalize_common_ocr_noise(digital))
    return "\n\n".join(parts).strip()


def extract_text_full_ocr(pdf_data: bytes, zoom: float = 2.25) -> str:
    """OCR every page (slow; use for heavily scanned PDFs)."""
    import fitz

    if not _ocr_available():
        text, _ = _best_digital_layer(pdf_data)
        return text

    parts: list[str] = []
    with fitz.open(stream=pdf_data, filetype="pdf") as doc:
        for page in doc:
            try:
                parts.append(_normalize_common_ocr_noise(_ocr_page_image(page, zoom=zoom)))
            except Exception:
                parts.append(
                    _normalize_common_ocr_noise(
                        page.get_text("text", sort=True) or ""
                    )
                )
    return "\n\n".join(parts).strip()


def pdf_bytes_to_text(
    pdf_data: bytes,
    *,
    use_ocr: bool = False,
    ocr_mode: str = "hybrid",
) -> str:
    """
    Convert PDF bytes to UTF-8 plain text.

    Parameters
    ----------
    use_ocr
        If False, only digital extraction (PyMuPDF + pdfminer heuristics).
    ocr_mode
        ``hybrid`` — OCR only pages with sparse digital text (recommended).
        ``full`` — OCR every page (slow, strongest for all-scan documents).
        ``off`` — same as ``use_ocr=False``.
    """
    if not use_ocr or ocr_mode == "off":
        text, _ = _best_digital_layer(pdf_data)
        return text

    if ocr_mode == "full":
        return extract_text_full_ocr(pdf_data)
    return extract_text_hybrid_per_page(pdf_data)


def pdf_to_text(pdf_data: bytes) -> str:
    """Backward-compatible entry: digital-only extraction."""
    return pdf_bytes_to_text(pdf_data, use_ocr=False)


def pdf_to_text_with_options(
    pdf_data: bytes,
    *,
    use_ocr: bool = False,
    ocr_mode: str = "hybrid",
) -> str:
    """Crawler entry with OCR flags."""
    return pdf_bytes_to_text(pdf_data, use_ocr=use_ocr, ocr_mode=ocr_mode)


def ocr_dependencies_ok() -> bool:
    """True if Tesseract + Python OCR bindings are usable."""
    return _ocr_available()
