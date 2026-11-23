"""Extract plain text from uploaded resumes (PDF, DOCX, TXT).

All parsing happens in memory from raw bytes so nothing sensitive is written to
disk. Every failure is raised as :class:`ResumeParseError` with a message that is
safe to show to the end user.
"""

from __future__ import annotations

import io
import os
import zipfile
from typing import Iterable

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB per resume
MAX_TEXT_CHARS = 200_000
MIN_TEXT_CHARS = 30


class ResumeParseError(ValueError):
    """Raised when a resume cannot be read or validated."""


def get_extension(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()


def validate_file(filename: str, data: bytes) -> str:
    """Validate name, size and magic bytes. Returns the normalised extension."""
    if not filename or not filename.strip():
        raise ResumeParseError("File has no name.")
    ext = get_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ResumeParseError(
            f"Unsupported file type '{ext or 'none'}'. Allowed types: {allowed}."
        )
    if not data:
        raise ResumeParseError("File is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise ResumeParseError(
            f"File is too large ({len(data) / 1024 / 1024:.1f} MB). "
            f"Maximum is {MAX_FILE_BYTES // 1024 // 1024} MB."
        )
    if ext == ".pdf" and not data.lstrip()[:5].startswith(b"%PDF"):
        raise ResumeParseError("File does not look like a valid PDF.")
    if ext == ".docx" and not zipfile.is_zipfile(io.BytesIO(data)):
        raise ResumeParseError("File does not look like a valid DOCX document.")
    return ext


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover - dependency problem
        raise ResumeParseError("PDF support requires the 'pypdf' package.") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise ResumeParseError("PDF is password protected.")
            except ResumeParseError:
                raise
            except Exception as exc:
                raise ResumeParseError("PDF is password protected.") from exc
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    except ResumeParseError:
        raise
    except PdfReadError as exc:
        raise ResumeParseError(f"Could not read PDF: {exc}") from exc
    except Exception as exc:
        raise ResumeParseError(f"Could not read PDF: {exc}") from exc


def _iter_docx_blocks(document) -> Iterable[str]:
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            yield paragraph.text
    for table in document.tables:
        for row in table.rows:
            cells = []
            for cell in row.cells:
                text = cell.text.strip()
                if text and text not in cells:
                    cells.append(text)
            if cells:
                yield " | ".join(cells)


def _extract_docx(data: bytes) -> str:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover - dependency problem
        raise ResumeParseError(
            "DOCX support requires the 'python-docx' package."
        ) from exc
    try:
        document = docx.Document(io.BytesIO(data))
        return "\n".join(_iter_docx_blocks(document))
    except Exception as exc:
        raise ResumeParseError(f"Could not read DOCX: {exc}") from exc


def _extract_txt(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeError:
            continue
    raise ResumeParseError("Could not decode text file.")  # pragma: no cover


def extract_text(filename: str, data: bytes) -> str:
    """Return cleaned text from a resume file or raise ResumeParseError."""
    ext = validate_file(filename, data)
    if ext == ".pdf":
        raw = _extract_pdf(data)
    elif ext == ".docx":
        raw = _extract_docx(data)
    else:
        raw = _extract_txt(data)

    raw = raw.replace("\x00", " ")
    text = "\n".join(line.rstrip() for line in raw.splitlines()).strip()
    if len(text) < MIN_TEXT_CHARS:
        raise ResumeParseError(
            "Could not extract enough text. The file may be a scanned image "
            "(OCR is not supported) or nearly empty."
        )
    return text[:MAX_TEXT_CHARS]
