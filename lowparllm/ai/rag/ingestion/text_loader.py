"""UTF-8 text, Markdown, HTML, and source-code extraction."""

from html.parser import HTMLParser
from pathlib import Path

from lowparllm.ai.rag.ingestion.base import (
    DocumentPage,
    DocumentProcessingError,
    ExtractedDocument,
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._blocked_depth = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        del _attrs
        if tag.lower() in {"script", "style"}:
            self._blocked_depth += 1
        elif tag.lower() in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self._blocked_depth:
            self._blocked_depth -= 1
        elif tag.lower() in {"p", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._blocked_depth:
            self.parts.append(data)


def load_text(path: Path, content: bytes) -> ExtractedDocument:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentProcessingError(
            "File is not valid UTF-8 text; no lossy conversion was attempted."
        ) from exc
    except OSError as exc:
        raise DocumentProcessingError(f"Could not read text file: {exc}") from exc
    if "\x00" in text:
        raise DocumentProcessingError("Binary content is not indexed as text.")
    if path.suffix.lower() == ".html":
        parser = _TextExtractor()
        try:
            parser.feed(text)
            parser.close()
        except ValueError as exc:
            raise DocumentProcessingError(f"Could not parse HTML: {exc}") from exc
        text = "".join(parser.parts)
    if not text.strip():
        raise DocumentProcessingError("No extractable text was found.")
    return ExtractedDocument(path=path, pages=(DocumentPage(text=text),))
