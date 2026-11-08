"""Single-process persistent Qdrant local vector store."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from qdrant_client import QdrantClient, models


class VectorStoreError(RuntimeError):
    """The local vector store is unavailable or incompatible."""


@dataclass(frozen=True)
class VectorHit:
    payload: dict[str, Any]
    score: float


class VectorStore:
    """Manage one Qdrant embedded client for the application process."""

    def __init__(self, path: Path, collection_name: str) -> None:
        self.path = path.expanduser()
        if self.path.is_symlink():
            raise VectorStoreError(
                f"Refusing to open a symbolic-link Qdrant directory: {self.path}"
            )
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        try:
            self._client = QdrantClient(path=str(self.path))
        except (OSError, RuntimeError, ValueError) as exc:
            raise VectorStoreError(
                "Could not open the local Qdrant database. It may be locked by "
                "another LowP process or need recovery."
            ) from exc
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._client.close()
            self._closed = True

    def collection_exists(self) -> bool:
        self._ensure_open()
        return self._client.collection_exists(self.collection_name)

    def collection_info(self) -> dict[str, Any]:
        self._ensure_open()
        if not self.collection_exists():
            return {"exists": False, "points": 0, "metadata": {}}
        info = self._client.get_collection(self.collection_name)
        metadata = getattr(info, "metadata", None)
        count = self._client.count(self.collection_name, exact=True).count
        return {
            "exists": True,
            "points": count,
            "metadata": metadata if isinstance(metadata, dict) else {},
        }

    def ensure_collection(self, model: str, dimensions: int) -> None:
        self._ensure_open()
        if dimensions < 1:
            raise VectorStoreError("Embedding vector dimensions must be positive.")
        if not self.collection_exists():
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dimensions, distance=models.Distance.COSINE
                ),
                metadata={
                    "embedding_model": model,
                    "vector_dimensions": dimensions,
                },
            )
            return

        info = self._client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            raise VectorStoreError(
                "The configured Qdrant collection uses named vectors, which LowP "
                "does not support."
            )
        stored_dimensions = getattr(vectors, "size", None)
        if stored_dimensions != dimensions:
            raise VectorStoreError(
                f"Embedding dimension mismatch: collection has {stored_dimensions}, "
                f"configured model returned {dimensions}."
            )
        metadata = getattr(info, "metadata", None)
        stored_model = metadata.get("embedding_model") if isinstance(metadata, dict) else None
        if stored_model and stored_model != model:
            raise VectorStoreError(
                f"Collection uses embedding model {stored_model!r}, not {model!r}. "
                "Use a separate collection or explicitly reset knowledge first."
            )
        if not stored_model:
            points, _ = self._client.scroll(
                collection_name=self.collection_name,
                limit=1,
                with_payload=["embedding_model", "vector_dimensions"],
                with_vectors=False,
            )
            if points:
                payload = points[0].payload or {}
                if payload.get("embedding_model") != model or payload.get(
                    "vector_dimensions"
                ) != dimensions:
                    raise VectorStoreError(
                        "Existing vectors do not have verifiable embedding metadata. "
                        "Use a separate collection or explicitly reset knowledge."
                    )

    def upsert(
        self,
        points: Sequence[models.PointStruct],
        *,
        model: str,
        dimensions: int,
    ) -> None:
        self._ensure_open()
        self.ensure_collection(model, dimensions)
        if not points:
            return
        for point in points:
            vector = point.vector
            if isinstance(vector, dict) or len(vector) != dimensions:
                raise VectorStoreError("A vector has an incompatible dimension.")
        self._client.upsert(
            collection_name=self.collection_name,
            points=list(points),
            wait=True,
        )

    def search(
        self,
        vector: list[float],
        *,
        limit: int,
        score_threshold: float,
    ) -> list[VectorHit]:
        self._ensure_open()
        if not self.collection_exists():
            return []
        info = self._client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict) or getattr(vectors, "size", None) != len(vector):
            raise VectorStoreError(
                "Query embedding dimensions do not match the stored collection."
            )
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        return [
            VectorHit(
                payload=point.payload if isinstance(point.payload, dict) else {},
                score=float(point.score),
            )
            for point in response.points
        ]

    def delete_document(self, source_id: str) -> None:
        self._ensure_open()
        if self.collection_exists():
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=source_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )

    def delete_old_versions(self, source_id: str, generation_id: str) -> None:
        self._ensure_open()
        if self.collection_exists():
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=source_id),
                            )
                        ],
                        must_not=[
                            models.FieldCondition(
                                key="generation_id",
                                match=models.MatchValue(value=generation_id),
                            )
                        ],
                    )
                ),
                wait=True,
            )

    def delete_points(self, point_ids: Sequence[str]) -> None:
        self._ensure_open()
        if point_ids and self.collection_exists():
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=list(point_ids)),
                wait=True,
            )

    def reset(self) -> None:
        self._ensure_open()
        if self.collection_exists():
            self._client.delete_collection(self.collection_name)

    def _ensure_open(self) -> None:
        if self._closed:
            raise VectorStoreError("The Qdrant client is already closed.")
