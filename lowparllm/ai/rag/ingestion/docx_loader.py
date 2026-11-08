"""Paragraph, heading, and table extraction for DOCX documents."""

from pathlib import Path
from io import BytesIO

from docx import Document
from docx.opc.exceptions import PackageNotFoundError

from lowparllm.ai.rag.ingestion.base import (
    DocumentPage,
    DocumentProcessingError,
    ExtractedDocument,
)


def load_docx(path: Path, content: bytes) -> ExtractedDocument:
    try:
        document = Document(BytesIO(content))
    except (OSError, PackageNotFoundError, ValueError) as exc:
        raise DocumentProcessingError(f"Could not read DOCX: {exc}") from exc
    lines: list[str] = []
    current_heading: str | None = None
    for paragraph in document.paragraphs:
        text = paragraph.text
        if not text.strip():
            continue
        if paragraph.style and paragraph.style.name.startswith("Heading"):
            current_heading = text.strip()
            lines.append(f"# {current_heading}")
        else:
            lines.append(text)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.replace("\n", " ").strip() for cell in row.cells]
            if any(cells):
                lines.append(" | ".join(cells))
    text = "\n".join(lines)
    if not text.strip():
        raise DocumentProcessingError("No extractable text was found in the DOCX.")
    return ExtractedDocument(
        path=path,
        pages=(DocumentPage(text=text, section_heading=current_heading),),
        metadata={"title": document.core_properties.title or ""},
    )
