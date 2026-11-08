"""Page-aware PDF text extraction without OCR."""

from pathlib import Path
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from lowparllm.ai.rag.ingestion.base import (
    DocumentPage,
    DocumentProcessingError,
    ExtractedDocument,
)


def load_pdf(path: Path, content: bytes) -> ExtractedDocument:
    try:
        reader = PdfReader(BytesIO(content), strict=False)
    except (OSError, PdfReadError, ValueError) as exc:
        raise DocumentProcessingError(f"Could not read PDF: {exc}") from exc
    if reader.is_encrypted:
        raise DocumentProcessingError(
            "This PDF is encrypted. Decrypt it yourself before importing."
        )
    pages: list[DocumentPage] = []
    try:
        for number, page in enumerate(reader.pages, start=1):
            if page.get("/Contents") is None:
                continue
            text = page.extract_text(extraction_mode="layout") or ""
            if text.strip():
                pages.append(DocumentPage(text=text, page_number=number))
    except (PdfReadError, ValueError) as exc:
        raise DocumentProcessingError(f"Could not extract PDF text: {exc}") from exc
    if not pages:
        raise DocumentProcessingError(
            "No extractable text was found. Scanned PDFs need OCR, which is not "
            "available in this phase."
        )
    metadata = reader.metadata
    return ExtractedDocument(
        path=path,
        pages=tuple(pages),
        metadata={
            "title": str(metadata.title) if metadata and metadata.title else "",
            "author": str(metadata.author) if metadata and metadata.author else "",
        },
    )
