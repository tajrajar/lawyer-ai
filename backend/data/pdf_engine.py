"""
PDF text extraction with layered fallbacks.

Tries fast text-layer extraction first (PyMuPDF, pdfplumber), then EasyOCR for
scanned documents, and appends table content extracted via pdfplumber.
"""

import re
import fitz
import pdfplumber
import logging

logger = logging.getLogger(__name__)

# Below this character count after normal extraction, assume a scanned PDF and retry with OCR.
OCR_RETRY_THRESHOLD = 200


def extract_text(pdf_path: str, use_ocr: bool = False) -> str:
    """
    Extract plain text from a PDF using a tiered extraction strategy.

    Args:
        pdf_path: Path to the PDF file on disk.
        use_ocr: When True, run EasyOCR if both PyMuPDF and pdfplumber yield little text.

    Returns:
        Cleaned extracted text, optionally including a [TABLES] section.
    """
    text = _pymupdf(pdf_path)

    if len(text.strip()) < 100:
        logger.info("Scanned PDF — trying pdfplumber")
        text = _pdfplumber(pdf_path)

    if len(text.strip()) < 100 and use_ocr:
        logger.info("Trying EasyOCR")
        text = _easyocr(pdf_path)

    tables = _extract_tables(pdf_path)
    if tables:
        text += "\n\n[TABLES]\n" + tables

    return _clean(text)


def extract_text_auto(pdf_path: str, ocr_threshold: int = OCR_RETRY_THRESHOLD) -> str:
    """
    Extract text and automatically enable OCR when the initial yield is very low.

    Scanned court filings often return near-empty text from digital extractors;
    this wrapper retries with EasyOCR when output falls below ocr_threshold chars.

    Args:
        pdf_path: Path to the PDF file.
        ocr_threshold: Character count below which OCR is attempted (default 200).

    Returns:
        Best available extracted text after optional OCR retry.
    """
    text = extract_text(pdf_path, use_ocr=False)
    if len(text.strip()) < ocr_threshold:
        logger.info(
            "Low text yield (%d chars) — retrying with OCR",
            len(text.strip()),
        )
        text = extract_text(pdf_path, use_ocr=True)
    return text


def _pymupdf(path: str) -> str:
    """Extract embedded text via PyMuPDF (fastest for digital PDFs)."""
    try:
        doc  = fitz.open(path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        logger.error(f"PyMuPDF: {e}")
        return ""


def _pdfplumber(path: str) -> str:
    """Fallback text extraction via pdfplumber (handles some layout-heavy PDFs)."""
    try:
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text
    except Exception as e:
        logger.error(f"pdfplumber: {e}")
        return ""


def _easyocr(path: str) -> str:
    """OCR each page as an image; supports English and Urdu script."""
    try:
        import easyocr
        from PIL import Image
        import io
        reader = easyocr.Reader(['en', 'ur'], gpu=False)
        doc    = fitz.open(path)
        text   = ""
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            res = reader.readtext(img)
            text += " ".join(r[1] for r in res) + "\n\n"
        doc.close()
        return text
    except Exception as e:
        logger.error(f"EasyOCR: {e}")
        return ""


def _extract_tables(path: str) -> str:
    """Serialize detected tables as pipe-delimited rows for LLM context."""
    try:
        out = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                for table in (page.extract_tables() or []):
                    for row in table:
                        if any(row):
                            out += " | ".join(str(c or "") for c in row) + "\n"
                    out += "\n"
        return out
    except Exception:
        return ""


def _clean(text: str) -> str:
    """Normalize whitespace and strip isolated page-number lines."""
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)
    return text.strip()
