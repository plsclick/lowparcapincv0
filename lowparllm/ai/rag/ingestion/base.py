"""Shared document metadata and parser error types."""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".docx",
    ".py",
    ".js",
    ".ts",
    ".java",
    ".c",
    ".cpp",
    ".json",
    ".html",
    ".css",
}

CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".java",
    ".c",
    ".cpp",
    ".json",
    ".css",
}


@dataclass(frozen=True)
class DocumentPage:
    text: str
    page_number: int | None = None
    section_heading: str | None = None


@dataclass(frozen=True)
class ExtractedDocument:
    path: Path
    pages: tuple[DocumentPage, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    file_size: int = 0


class DocumentProcessingError(ValueError):
    """A document could not be safely or meaningfully extracted."""


def extract_document(path: Path, max_file_size_mb: int = 50) -> ExtractedDocument:
    """Dispatch a supported local document to its format-specific parser."""
    from lowparllm.ai.rag.ingestion.docx_loader import load_docx
    from lowparllm.ai.rag.ingestion.pdf_loader import load_pdf
    from lowparllm.ai.rag.ingestion.text_loader import load_text

    candidate = path.expanduser()
    if candidate.is_symlink():
        raise DocumentProcessingError("Symbolic-link documents are not indexed.")
    try:
        resolved = candidate.resolve(strict=True)
        details = resolved.stat()
    except OSError as exc:
        raise DocumentProcessingError(f"Could not access document: {exc}") from exc
    if not resolved.is_file():
        raise DocumentProcessingError("The selected path is not a regular file.")
    max_bytes = max_file_size_mb * 1024 * 1024
    if details.st_size > max_bytes:
        raise DocumentProcessingError(
            f"File is {details.st_size / 1024 / 1024:.1f} MiB; the configured "
            f"limit is {max_file_size_mb} MiB."
        )
    if details.st_size == 0:
        raise DocumentProcessingError("The selected file is empty.")
    try:
        raw_data = resolved.read_bytes()
    except OSError as exc:
        raise DocumentProcessingError(f"Could not read document: {exc}") from exc
    if len(raw_data) > max_bytes:
        raise DocumentProcessingError("The document exceeded its size limit while reading.")
    suffix = resolved.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentProcessingError(
            f"Unsupported file type {suffix or '(no extension)'!r}."
        )
    if suffix == ".pdf":
        document = load_pdf(resolved, raw_data)
    elif suffix == ".docx":
        document = load_docx(resolved, raw_data)
    else:
        document = load_text(resolved, raw_data)
    return ExtractedDocument(
        path=document.path,
        pages=document.pages,
        metadata=document.metadata,
        content_hash=hashlib.sha256(raw_data).hexdigest(),
        file_size=len(raw_data),
    )
