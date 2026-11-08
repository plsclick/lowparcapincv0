"""Embedding-backed semantic retrieval with active-document validation."""

import asyncio
from dataclasses import dataclass

from lowparllm.ai.embeddings.ollama_embeddings import OllamaEmbeddings
from lowparllm.ai.rag.vector_store import VectorStore
from lowparllm.storage.database import DocumentRegistry


class RetrievalError(RuntimeError):
    """A local knowledge-base search could not be completed."""


@dataclass(frozen=True)
class RetrievedChunk:
    payload: dict[str, object]
    score: float


class Retriever:
    def __init__(
        self,
        embeddings: OllamaEmbeddings,
        vectors: VectorStore,
        registry: DocumentRegistry,
        embedding_model: str,
    ) -> None:
        self.embeddings = embeddings
        self.vectors = vectors
        self.registry = registry
        self.embedding_model = embedding_model

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        score_threshold: float = 0.35,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between zero and one.")
        if self.registry.indexed_source_count() == 0:
            return []

        metadata = self.registry.collection_metadata()
        stored_model = metadata.get("embedding_model")
        if stored_model and stored_model != self.embedding_model:
            raise RetrievalError(
                f"The knowledge base uses {stored_model!r}, but the configured "
                f"embedding model is {self.embedding_model!r}."
            )
        try:
            vectors = await self.embeddings.embed([query])
            if not vectors:
                return []
            dimensions = len(vectors[0])
            stored_dimensions = metadata.get("vector_dimensions")
            if stored_dimensions and int(stored_dimensions) != dimensions:
                raise RetrievalError(
                    "The query embedding dimensions do not match the knowledge base."
                )
            hits = await asyncio.to_thread(
                self.vectors.search,
                vectors[0],
                limit=min(top_k * 4, 200),
                score_threshold=score_threshold,
            )
        except RetrievalError:
            raise
        except (RuntimeError, ValueError, OSError) as exc:
            raise RetrievalError(f"Local semantic search failed: {exc}") from exc

        source_ids = list(
            {
                str(hit.payload["document_id"])
                for hit in hits
                if hit.payload.get("document_id")
            }
        )
        current_sources = self.registry.current_sources(source_ids)
        selected: list[RetrievedChunk] = []
        seen_chunks: set[tuple[str, str]] = set()
        for hit in hits:
            source_id = hit.payload.get("document_id")
            file_hash = hit.payload.get("file_hash")
            chunk_hash = hit.payload.get("content_hash")
            source = current_sources.get(source_id) if isinstance(source_id, str) else None
            if (
                source is None
                or source.content_hash != file_hash
                or hit.payload.get("embedding_model") != self.embedding_model
                or not isinstance(chunk_hash, str)
            ):
                continue
            if not isinstance(hit.payload.get("text"), str):
                continue
            chunk_key = (str(source_id), chunk_hash)
            if chunk_key in seen_chunks:
                continue
            seen_chunks.add(chunk_key)
            payload = dict(hit.payload)
            payload["filename"] = source.filename
            payload["file_type"] = source.file_type
            payload["source_path"] = source.original_path
            selected.append(RetrievedChunk(payload, hit.score))
            if len(selected) >= top_k:
                break
        return selected
