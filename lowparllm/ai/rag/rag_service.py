"""Local document indexing, retrieval, lifecycle, and knowledge-base status."""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import sqlite3
import tempfile
import uuid
from collections.abc import Callable
from typing import Any

from qdrant_client import models

from lowparllm.ai.embeddings.ollama_embeddings import (
    EmbeddingError,
    OllamaEmbeddings,
)
from lowparllm.ai.rag.citations import Citation, build_references
from lowparllm.ai.rag.document_processor import DocumentProcessor
from lowparllm.ai.rag.ingestion.base import DocumentProcessingError
from lowparllm.ai.rag.ingestion.directory_loader import discover_documents
from lowparllm.ai.rag.retriever import RetrievedChunk, Retriever
from lowparllm.ai.rag.vector_store import VectorStore, VectorStoreError
from lowparllm.core.config import (
    HARD_MAX_WEB_RESPONSE_BYTES,
    AppConfig,
    persist_rag_automatic_retrieval,
)
from lowparllm.storage.database import (
    ChunkRecord,
    DocumentRegistry,
    SourceRecord,
)

logger = logging.getLogger(__name__)


def _write_web_source(
    data_root: Path,
    title: str,
    metadata: dict[str, Any],
    content: str,
) -> Path:
    root = data_root.resolve()
    directory = data_root / "web-sources"
    if directory.is_symlink():
        raise RagServiceError("Refusing to save a web source through a symbolic link.")
    directory.mkdir(parents=True, exist_ok=True)
    resolved_directory = directory.resolve()
    if not resolved_directory.is_relative_to(root):
        raise RagServiceError("The web-source directory must remain inside LowP data.")

    url = str(metadata["url"])
    filename = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32] + ".md"
    target = directory / filename
    if target.is_symlink():
        raise RagServiceError("Refusing to overwrite a symbolic-link web source.")
    marker = f"<!-- LowP saved web source: {url} -->"
    if target.exists():
        try:
            with target.open("r", encoding="utf-8") as existing_file:
                existing_marker = existing_file.readline().rstrip("\r\n")
        except (OSError, UnicodeError) as exc:
            raise RagServiceError(
                "Refusing to overwrite an existing file in the web-source directory."
            ) from exc
        if existing_marker != marker:
            raise RagServiceError(
                "Refusing to overwrite a file not created by LowP in the web-source directory."
            )
    heading = " ".join(title.split()) or metadata["domain"]
    lines = [
        marker,
        f"# {heading}",
        "",
        f"Source URL: {url}",
        *(
            [f"Retrieved URL: {metadata['final_url']}"]
            if metadata.get("final_url") and metadata["final_url"] != url
            else []
        ),
        *(
            [f"Canonical URL: {metadata['canonical_url']}"]
            if metadata.get("canonical_url")
            and metadata["canonical_url"] not in {url, metadata.get("final_url")}
            else []
        ),
        f"Domain: {metadata['domain']}",
        f"Published: {metadata['published_date'] or 'Not provided'}",
        f"Accessed: {metadata['accessed_at']}",
        "",
        content,
        "",
    ]
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".web-source.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write("\n".join(lines))
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, target)
    except OSError:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return target


@dataclass(frozen=True)
class IngestionResult:
    source_id: str | None
    status: str
    message: str


@dataclass(frozen=True)
class SearchResult:
    chunks: tuple[RetrievedChunk, ...]


@dataclass(frozen=True)
class ChatContext:
    text: str
    citations: tuple[Citation, ...]
    chunks: tuple[RetrievedChunk, ...]


class RagServiceError(RuntimeError):
    """A local knowledge-base operation could not be completed."""


class RagService:
    """Own one registry, one local Qdrant client, and one embedding client."""

    def __init__(
        self,
        config: AppConfig,
        *,
        embeddings: OllamaEmbeddings | None = None,
    ) -> None:
        self.config = config
        rag = config.rag
        self.embeddings = embeddings or OllamaEmbeddings(
            config.ollama_url,
            rag.embedding_model,
            batch_size=rag.embedding_batch_size,
        )
        self._owns_embeddings = embeddings is None
        self._registry: DocumentRegistry | None = None
        self._vectors: VectorStore | None = None
        self._retriever: Retriever | None = None
        self._initialization_lock = asyncio.Lock()
        self._automatic_retrieval = rag.enabled and rag.automatic_retrieval
        self._cache: OrderedDict[tuple[int, str], ChatContext] = OrderedDict()
        self._cache_revision = 0

    @property
    def automatic_retrieval(self) -> bool:
        return self._automatic_retrieval

    async def close(self) -> None:
        try:
            if self._vectors is not None:
                await asyncio.to_thread(self._vectors.close)
        finally:
            self._vectors = None
            self._retriever = None
            try:
                if self._registry is not None:
                    self._registry.close()
            finally:
                self._registry = None
                if self._owns_embeddings:
                    await self.embeddings.aclose()

    async def set_automatic_retrieval(self, enabled: bool) -> None:
        if enabled and not self.config.rag.enabled:
            raise RagServiceError(
                "RAG is disabled in configuration. Set [rag].enabled = true first."
            )
        persist_rag_automatic_retrieval(enabled)
        self._automatic_retrieval = enabled and self.config.rag.enabled
        self._cache.clear()

    async def status(self) -> dict[str, Any]:
        try:
            await self._ensure_initialized()
            assert self._registry is not None and self._vectors is not None
            info = await asyncio.to_thread(self._vectors.collection_info)
            try:
                embedding_available = await self.embeddings.is_available(refresh=True)
                embedding_error = None
                embedding_model_missing = not embedding_available
            except EmbeddingError as exc:
                embedding_available = False
                embedding_error = str(exc)
                embedding_model_missing = False
            metadata = self._registry.collection_metadata()
            return {
                "qdrant_available": True,
                "sqlite_available": True,
                "indexed_documents": self._registry.indexed_source_count(),
                "sources": self._registry.source_count(),
                "chunks": self._registry.chunk_count(),
                "vectors": info["points"],
                "collection_exists": info["exists"],
                "collection_name": self.config.rag.collection_name,
                "embedding_model": self.config.rag.embedding_model,
                "embedding_model_available": embedding_available,
                "embedding_model_missing": embedding_model_missing,
                "embedding_error": embedding_error,
                "vector_dimensions": metadata.get("vector_dimensions"),
                "qdrant_path": str(self.config.rag.qdrant_path),
                "sqlite_path": str(self.config.rag.sqlite_path),
                "enabled": self.config.rag.enabled,
                "automatic_retrieval": self._automatic_retrieval,
            }
        except (
            RuntimeError,
            sqlite3.Error,
            OSError,
            ValueError,
        ) as exc:
            return {
                "qdrant_available": False,
                "sqlite_available": self._registry is not None,
                "indexed_documents": 0,
                "sources": 0,
                "chunks": 0,
                "vectors": 0,
                "collection_exists": False,
                "collection_name": self.config.rag.collection_name,
                "embedding_model": self.config.rag.embedding_model,
                "embedding_model_available": False,
                "embedding_model_missing": False,
                "embedding_error": str(exc),
                "vector_dimensions": None,
                "qdrant_path": str(self.config.rag.qdrant_path),
                "sqlite_path": str(self.config.rag.sqlite_path),
                "enabled": self.config.rag.enabled,
                "automatic_retrieval": self._automatic_retrieval,
            }

    async def add_file(
        self,
        path: Path,
        *,
        force: bool = False,
        source_metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        await self._ensure_initialized()
        assert self._registry is not None and self._vectors is not None
        candidate = path.expanduser()
        existing_path = (
            None
            if candidate.is_symlink()
            else self._registry.find_path(str(candidate.resolve()))
        )
        try:
            document, chunks = await DocumentProcessor(
                self.config.rag
            ).process_async(candidate)
        except (DocumentProcessingError, OSError, ValueError) as exc:
            self._record_processing_failure(candidate, exc, existing_path)
            return IngestionResult(
                existing_path.id if existing_path else None, "failed", str(exc)
            )

        duplicate = self._registry.find_hash(document.content_hash)
        if duplicate is not None and not force:
            if duplicate.original_path != str(document.path):
                duplicate = self._registry.update_path(
                    duplicate.id, document.path, document.file_size
                )
                self._invalidate_cache()
            return IngestionResult(
                duplicate.id, "skipped", "Unchanged content is already indexed."
            )

        source_id = (
            existing_path.id
            if existing_path is not None
            else duplicate.id
            if force and duplicate is not None
            else str(uuid.uuid4())
        )
        try:
            embedding_available = await self.embeddings.is_available()
        except EmbeddingError as exc:
            self._registry.record_failure(
                source_id=source_id,
                path=document.path,
                file_type=document.path.suffix.lower(),
                file_size=document.file_size,
                file_hash=document.content_hash,
                error=str(exc),
            )
            return IngestionResult(source_id, "failed", str(exc))
        if not embedding_available:
            message = (
                f"Embedding model {self.config.rag.embedding_model!r} is not "
                f"installed. Install it manually with: ollama pull "
                f"{self.config.rag.embedding_model}. LowP will not download it."
            )
            self._registry.record_failure(
                source_id=source_id,
                path=document.path,
                file_type=document.path.suffix.lower(),
                file_size=document.file_size,
                file_hash=document.content_hash,
                error=message,
            )
            return IngestionResult(source_id, "failed", message)
        if not chunks:
            message = "Document extraction produced no indexable text chunks."
            self._registry.record_failure(
                source_id=source_id,
                path=document.path,
                file_type=document.path.suffix.lower(),
                file_size=document.file_size,
                file_hash=document.content_hash,
                error=message,
            )
            return IngestionResult(source_id, "failed", message)

        stored_metadata = self._registry.collection_metadata()
        stored_model = stored_metadata.get("embedding_model")
        if stored_model and stored_model != self.config.rag.embedding_model:
            message = (
                f"The knowledge base uses embedding model {stored_model!r}, not "
                f"{self.config.rag.embedding_model!r}. Use a separate collection "
                "or explicitly reset knowledge first."
            )
            self._registry.record_failure(
                source_id=source_id,
                path=document.path,
                file_type=document.path.suffix.lower(),
                file_size=document.file_size,
                file_hash=document.content_hash,
                error=message,
            )
            return IngestionResult(source_id, "failed", message)

        uploaded_ids: list[str] = []
        chunk_records: list[ChunkRecord] = []
        vector_dimensions: int | None = None
        generation_id = uuid.uuid4().hex
        pending_points: list[models.PointStruct] = []
        try:
            texts = [chunk.text for chunk in chunks]
            cursor = 0
            async for vector_batch in self.embeddings.embed_batches(texts):
                batch_chunks = chunks[cursor : cursor + len(vector_batch)]
                if len(batch_chunks) != len(vector_batch):
                    raise RagServiceError("Embedding batch size did not match document chunks.")
                dimensions = len(vector_batch[0]) if vector_batch else 0
                if vector_dimensions is None:
                    vector_dimensions = dimensions
                    if (
                        stored_metadata.get("vector_dimensions")
                        and int(stored_metadata["vector_dimensions"]) != dimensions
                    ):
                        raise RagServiceError(
                            "Embedding dimensions do not match the existing collection."
                        )
                elif vector_dimensions != dimensions:
                    raise RagServiceError("Embedding dimensions changed during indexing.")

                for chunk, vector in zip(batch_chunks, vector_batch, strict=True):
                    point_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"lowp:{source_id}:{document.content_hash}:"
                            f"{generation_id}:{chunk.chunk_index}",
                        )
                    )
                    uploaded_ids.append(point_id)
                    chunk_records.append(
                        ChunkRecord(
                            chunk_id=point_id,
                            chunk_index=chunk.chunk_index,
                            content_hash=chunk.content_hash,
                            page_number=chunk.page_number,
                            section_heading=chunk.section_heading,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                        )
                    )
                    pending_points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={
                                "document_id": source_id,
                                "file_hash": document.content_hash,
                                "content_hash": chunk.content_hash,
                                "generation_id": generation_id,
                                "embedding_model": self.config.rag.embedding_model,
                                "vector_dimensions": dimensions,
                                "filename": document.path.name,
                                "source_path": str(document.path),
                                "file_type": document.path.suffix.lower(),
                                "page_number": chunk.page_number,
                                "section_heading": chunk.section_heading,
                                "start_line": chunk.start_line,
                                "end_line": chunk.end_line,
                                "chunk_index": chunk.chunk_index,
                                "text": chunk.text,
                                "source_metadata": source_metadata or {},
                            },
                        )
                    )
                if len(pending_points) >= max(
                    16, self.config.rag.embedding_batch_size * 4
                ):
                    await self._upsert_safely(pending_points, dimensions)
                    pending_points.clear()
                cursor += len(batch_chunks)

            if pending_points and vector_dimensions is not None:
                await self._upsert_safely(pending_points, vector_dimensions)
                pending_points.clear()
            if cursor != len(chunks) or vector_dimensions is None:
                raise RagServiceError("Embedding generation returned incomplete results.")
            self._registry.register_indexed(
                source_id=source_id,
                path=document.path,
                file_type=document.path.suffix.lower(),
                file_size=document.file_size,
                file_hash=document.content_hash,
                chunks=chunk_records,
                embedding_model=self.config.rag.embedding_model,
                metadata={
                    **document.metadata,
                    **({"web_source": source_metadata} if source_metadata else {}),
                    "vector_dimensions": vector_dimensions,
                },
            )
        except asyncio.CancelledError:
            if uploaded_ids:
                try:
                    await asyncio.to_thread(self._vectors.delete_points, uploaded_ids)
                except VectorStoreError:
                    logger.exception("Could not clean up cancelled document vectors.")
            raise
        except (
            RuntimeError,
            sqlite3.Error,
            OSError,
            ValueError,
        ) as exc:
            if uploaded_ids:
                try:
                    await asyncio.to_thread(self._vectors.delete_points, uploaded_ids)
                except VectorStoreError:
                    logger.exception("Could not roll back incomplete vector points.")
            try:
                self._registry.record_failure(
                    source_id=source_id,
                    path=document.path,
                    file_type=document.path.suffix.lower(),
                    file_size=document.file_size,
                    file_hash=document.content_hash,
                    error=str(exc),
                )
            except sqlite3.Error:
                logger.exception("Could not record failed document indexing.")
            return IngestionResult(source_id, "failed", str(exc))

        try:
            await asyncio.to_thread(
                self._vectors.delete_old_versions, source_id, generation_id
            )
        except (RuntimeError, OSError, ValueError):
            logger.exception(
                "The new index is committed, but obsolete Qdrant points remain; "
                "retrieval filters points against the active content hash."
            )
        self._invalidate_cache()
        return IngestionResult(source_id, "indexed", f"Indexed {len(chunks)} chunks.")

    async def _upsert_safely(
        self, points: list[models.PointStruct], dimensions: int
    ) -> None:
        assert self._vectors is not None
        task = asyncio.create_task(
            asyncio.to_thread(
                self._vectors.upsert,
                points,
                model=self.config.rag.embedding_model,
                dimensions=dimensions,
            )
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await task
            except Exception:
                logger.exception("Qdrant write failed while indexing was cancelled.")
            raise

    async def add_folder(
        self,
        path: Path,
        progress: Callable[[int, int, Path], None] | None = None,
    ) -> tuple[list[Path], list[tuple[Path, IngestionResult]]]:
        files = await asyncio.to_thread(
            discover_documents,
            path,
            recursive=self.config.rag.recursive,
            ignore_hidden=self.config.rag.ignore_hidden,
            exclude_patterns=self.config.rag.exclude_patterns,
        )
        results: list[tuple[Path, IngestionResult]] = []
        for index, file_path in enumerate(files, start=1):
            if progress is not None:
                progress(index, len(files), file_path)
            results.append((file_path, await self.add_file(file_path)))
        return files, results

    async def add_web_source(
        self,
        *,
        title: str,
        url: str,
        domain: str,
        published_date: str | None,
        accessed_at: str,
        content: str,
        final_url: str | None = None,
        canonical_url: str | None = None,
    ) -> IngestionResult:
        if not content.strip():
            raise RagServiceError("The web source has no readable content to save.")
        if self.config.web.unlimited:
            if len(content.encode("utf-8")) > HARD_MAX_WEB_RESPONSE_BYTES:
                raise RagServiceError(
                    "The web source exceeds the hard response-size safety ceiling."
                )
        elif len(content) > self.config.web.max_page_chars:
            raise RagServiceError("The web source exceeds the configured page-text limit.")
        source_metadata = {
            "url": url,
            "domain": domain,
            "published_date": published_date,
            "accessed_at": accessed_at,
            "final_url": final_url,
            "canonical_url": canonical_url,
        }
        source_path = await asyncio.to_thread(
            _write_web_source,
            self.config.rag.sqlite_path.parent,
            title,
            source_metadata,
            content,
        )
        return await self.add_file(source_path, source_metadata=source_metadata)

    async def reindex(self, source_id: str) -> IngestionResult:
        await self._ensure_initialized()
        source = self._require_source(source_id)
        path = Path(source.original_path)
        if not path.is_file():
            return IngestionResult(
                source.id, "failed", "The original source file is no longer available."
            )
        return await self.add_file(path, force=True)

    async def list_sources(
        self, page: int = 1, page_size: int = 20
    ) -> tuple[list[SourceRecord], int]:
        await self._ensure_initialized()
        assert self._registry is not None
        return (
            self._registry.list_sources(page_size, (page - 1) * page_size),
            self._registry.source_count(),
        )

    async def get_source(self, source_id: str) -> SourceRecord | None:
        await self._ensure_initialized()
        assert self._registry is not None
        return self._registry.get_source(source_id)

    async def search(self, query: str) -> SearchResult:
        await self._ensure_initialized()
        assert self._retriever is not None
        chunks = await self._retriever.search(
            query,
            top_k=self.config.rag.top_k,
            score_threshold=self.config.rag.score_threshold,
        )
        return SearchResult(tuple(chunks))

    async def context_for_chat(self, query: str) -> ChatContext:
        if not self.config.rag.enabled or not self._automatic_retrieval:
            return ChatContext("", (), ())
        key = (self._cache_revision, query)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        search = await self.search(query)
        context, citations, used = self._bounded_context(search.chunks)
        result = ChatContext(context, citations, used)
        self._cache[key] = result
        if len(self._cache) > 32:
            self._cache.popitem(last=False)
        return result

    async def has_indexed_sources(self) -> bool:
        if not self.config.rag.enabled or not self._automatic_retrieval:
            return False
        await self._ensure_initialized()
        assert self._registry is not None
        return self._registry.indexed_source_count() > 0

    async def remove_source(self, source_id: str) -> SourceRecord:
        await self._ensure_initialized()
        assert self._registry is not None and self._vectors is not None
        source = self._require_source(source_id)
        self._registry.mark_removing(source.id)
        await asyncio.to_thread(self._vectors.delete_document, source.id)
        self._registry.finish_removing(source.id)
        self._invalidate_cache()
        return source

    async def reset_knowledge(self) -> None:
        await self._ensure_initialized()
        assert self._registry is not None and self._vectors is not None
        self._registry.mark_reset_pending()
        await asyncio.to_thread(self._vectors.reset)
        self._registry.clear_sources()
        self._invalidate_cache()

    async def _ensure_initialized(self) -> None:
        if self._registry is not None and self._vectors is not None:
            return
        async with self._initialization_lock:
            if self._registry is not None and self._vectors is not None:
                return
            registry: DocumentRegistry | None = None
            vectors: VectorStore | None = None
            try:
                registry = DocumentRegistry(self.config.rag.sqlite_path)
                vectors = await asyncio.to_thread(
                    VectorStore,
                    self.config.rag.qdrant_path,
                    self.config.rag.collection_name,
                )
                self._registry = registry
                self._vectors = vectors
                self._retriever = Retriever(
                    self.embeddings,
                    vectors,
                    registry,
                    self.config.rag.embedding_model,
                )
                if registry.reset_pending():
                    await asyncio.to_thread(vectors.reset)
                    registry.clear_sources()
                for source_id in registry.sources_to_recover():
                    await asyncio.to_thread(vectors.delete_document, source_id)
                    registry.finish_removing(source_id)
            except (RuntimeError, sqlite3.Error, OSError, ValueError) as exc:
                if vectors is not None:
                    await asyncio.to_thread(vectors.close)
                if registry is not None:
                    registry.close()
                self._registry = None
                self._vectors = None
                self._retriever = None
                raise RagServiceError(f"Could not initialize the local knowledge base: {exc}") from exc

    def _require_source(self, source_id: str) -> SourceRecord:
        assert self._registry is not None
        source = self._registry.get_source(source_id)
        if source is None:
            raise RagServiceError(
                f"Source {source_id!r} was not found or the ID prefix is ambiguous."
            )
        return source

    def _record_processing_failure(
        self, path: Path, error: Exception, source: SourceRecord | None
    ) -> None:
        assert self._registry is not None
        if path.is_symlink():
            resolved_path = path.absolute()
            self._registry.record_failure(
                source_id=str(uuid.uuid4()),
                path=resolved_path,
                file_type=path.suffix.lower(),
                file_size=0,
                file_hash=hashlib.sha256(str(resolved_path).encode("utf-8")).hexdigest(),
                error=str(error),
            )
            return
        try:
            details = path.stat()
            file_size = details.st_size
            resolved_path = path.resolve()
            file_type = resolved_path.suffix.lower()
            marker = f"{resolved_path}:{details.st_size}:{details.st_mtime_ns}"
        except OSError:
            file_size = 0
            resolved_path = path.absolute()
            file_type = path.suffix.lower()
            marker = str(resolved_path)
        self._registry.record_failure(
            source_id=source.id if source else str(uuid.uuid4()),
            path=resolved_path,
            file_type=file_type,
            file_size=file_size,
            file_hash=hashlib.sha256(marker.encode("utf-8")).hexdigest(),
            error=str(error),
        )

    def _invalidate_cache(self) -> None:
        self._cache_revision += 1
        self._cache.clear()

    def _bounded_context(
        self, chunks: tuple[RetrievedChunk, ...]
    ) -> tuple[str, tuple[Citation, ...], tuple[RetrievedChunk, ...]]:
        context_tokens = min(
            self.config.rag.max_context_tokens,
            max(128, self.config.context_size // 2),
        )
        max_characters = context_tokens * 4
        selected: list[RetrievedChunk] = []
        context = ""
        citations: tuple[Citation, ...] = ()
        for result in chunks:
            selected.append(result)
            candidate_context, candidate_citations = build_references(selected)
            if len(candidate_context) <= max_characters:
                context, citations = candidate_context, candidate_citations
                continue
            selected.pop()
            if not selected:
                payload = dict(result.payload)
                source_text = str(payload.get("text", ""))
                low, high = 0, len(source_text)
                best_context = ""
                best_citations: tuple[Citation, ...] = ()
                while low <= high:
                    middle = (low + high) // 2
                    payload["text"] = source_text[:middle]
                    candidate_context, candidate_citations = build_references(
                        [RetrievedChunk(payload, result.score)]
                    )
                    if len(candidate_context) <= max_characters:
                        best_context = candidate_context
                        best_citations = candidate_citations
                        low = middle + 1
                    else:
                        high = middle - 1
                selected.append(RetrievedChunk(payload, result.score))
                context, citations = best_context, best_citations
            break
        return context, citations, tuple(selected)
