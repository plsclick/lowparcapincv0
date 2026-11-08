"""Bounded file processing and stable content hashing before embedding."""

import asyncio
from pathlib import Path

from lowparllm.ai.rag.chunking import DocumentChunk, chunk_document
from lowparllm.ai.rag.ingestion.base import ExtractedDocument, extract_document
from lowparllm.core.config import RagConfig


class DocumentProcessor:
    def __init__(self, config: RagConfig) -> None:
        self.config = config

    def process(self, path: Path) -> tuple[ExtractedDocument, tuple[DocumentChunk, ...]]:
        document = extract_document(path, self.config.max_file_size_mb)
        chunks = chunk_document(
            document,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        return document, chunks

    async def process_async(
        self, path: Path
    ) -> tuple[ExtractedDocument, tuple[DocumentChunk, ...]]:
        return await asyncio.to_thread(self.process, path)
