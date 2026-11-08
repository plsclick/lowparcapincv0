"""Predictable paragraph- and line-aware document chunking."""

from dataclasses import dataclass
import hashlib
import re
from collections.abc import Iterator

from lowparllm.ai.rag.ingestion.base import CODE_EXTENSIONS, ExtractedDocument


_WORD = re.compile(r"\S+")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    chunk_index: int
    page_number: int | None
    section_heading: str | None
    start_line: int
    end_line: int
    content_hash: str


@dataclass(frozen=True)
class _Unit:
    text: str
    start_line: int
    end_line: int
    heading: str | None
    page_number: int | None


def _word_count(text: str) -> int:
    return sum(1 for _ in _WORD.finditer(text))


def _split_long_unit(unit: _Unit, max_words: int) -> list[_Unit]:
    if _word_count(unit.text) <= max_words:
        return [unit]
    sentences = re.split(r"(?<=[.!?])\s+", unit.text)
    if len(sentences) > 1:
        parts: list[_Unit] = []
        current: list[str] = []
        current_words = 0
        for sentence in sentences:
            sentence_words = _word_count(sentence)
            if current and current_words + sentence_words > max_words:
                parts.append(
                    _Unit(
                        "\n".join(current),
                        unit.start_line,
                        unit.end_line,
                        unit.heading,
                        unit.page_number,
                    )
                )
                current = []
                current_words = 0
            if sentence_words > max_words:
                if current:
                    parts.append(
                        _Unit(
                            "\n".join(current),
                            unit.start_line,
                            unit.end_line,
                            unit.heading,
                            unit.page_number,
                        )
                    )
                    current = []
                    current_words = 0
                sentence_unit = _Unit(
                    sentence,
                    unit.start_line,
                    unit.end_line,
                    unit.heading,
                    unit.page_number,
                )
                parts.extend(_split_long_unit(sentence_unit, max_words))
            else:
                current.append(sentence)
                current_words += sentence_words
        if current:
            parts.append(
                _Unit(
                    "\n".join(current),
                    unit.start_line,
                    unit.end_line,
                    unit.heading,
                    unit.page_number,
                )
            )
        return parts

    matches = list(_WORD.finditer(unit.text))
    parts: list[_Unit] = []
    for offset in range(0, len(matches), max_words):
        selected = matches[offset : offset + max_words]
        text = unit.text[selected[0].start() : selected[-1].end()]
        parts.append(
            _Unit(
                text=text,
                start_line=unit.start_line,
                end_line=unit.end_line,
                heading=unit.heading,
                page_number=unit.page_number,
            )
        )
    return parts


def _page_units(
    text: str,
    heading: str | None,
    page_number: int | None,
    code: bool,
) -> Iterator[_Unit]:
    lines = text.splitlines()
    if not lines:
        return
    if code:
        for number, line in enumerate(lines, start=1):
            if line.strip():
                yield _Unit(line, number, number, heading, page_number)
        return

    block: list[str] = []
    block_start = 1
    current_heading = heading
    for number, line in enumerate(lines, start=1):
        match = _HEADING.match(line)
        if match:
            if block:
                yield _Unit(
                    "\n".join(block), block_start, number - 1, current_heading, page_number
                )
                block.clear()
            current_heading = match.group(1).strip()
            yield _Unit(line.strip(), number, number, current_heading, page_number)
        elif not line.strip():
            if block:
                yield _Unit(
                    "\n".join(block), block_start, number - 1, current_heading, page_number
                )
                block.clear()
        else:
            if not block:
                block_start = number
            block.append(line)
    if block:
        yield _Unit(
            "\n".join(block), block_start, len(lines), current_heading, page_number
        )


def chunk_document(
    document: ExtractedDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> tuple[DocumentChunk, ...]:
    """Split extracted pages using whitespace-word budgets and complete units."""
    if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")
    units: list[_Unit] = []
    code = document.path.suffix.lower() in CODE_EXTENSIONS
    for page in document.pages:
        for unit in _page_units(
            page.text, page.section_heading, page.page_number, code
        ):
            units.extend(_split_long_unit(unit, chunk_size))
    if not units:
        return ()

    results: list[DocumentChunk] = []
    start = 0
    while start < len(units):
        end = start
        words = 0
        while end < len(units):
            if (
                end > start
                and units[end].page_number != units[start].page_number
            ):
                break
            next_words = _word_count(units[end].text)
            if end > start and words + next_words > chunk_size:
                break
            words += next_words
            end += 1
        if end == start:
            end += 1
        selected = units[start:end]
        text = "\n\n".join(unit.text for unit in selected).strip()
        if text:
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            results.append(
                DocumentChunk(
                    text=text,
                    chunk_index=len(results),
                    page_number=selected[0].page_number,
                    section_heading=next(
                        (unit.heading for unit in selected if unit.heading), None
                    ),
                    start_line=selected[0].start_line,
                    end_line=selected[-1].end_line,
                    content_hash=content_hash,
                )
            )
        if end >= len(units):
            break
        if chunk_overlap == 0:
            start = end
            continue
        retained = 0
        next_start = end
        while next_start > start + 1:
            candidate_words = _word_count(units[next_start - 1].text)
            if retained + candidate_words > chunk_overlap:
                break
            next_start -= 1
            retained += candidate_words
        start = max(start + 1, next_start)
    return tuple(results)
