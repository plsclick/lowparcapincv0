"""Local batched embeddings through Ollama's dedicated embedding API."""

import logging
import math
from collections.abc import AsyncIterator, Sequence

import httpx


logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Base error for local embedding failures."""


class EmbeddingModelMissing(EmbeddingError):
    """The configured local embedding model is not installed."""


class EmbeddingProtocolError(EmbeddingError):
    """Ollama returned an invalid embedding response."""


class OllamaEmbeddings:
    """Generate embeddings locally without invoking the chat model."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        batch_size: int = 16,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=5.0),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
        )
        self._owns_client = client is None
        self._available: bool | None = None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def is_available(self, *, refresh: bool = False) -> bool:
        if self._available is not None and not refresh:
            return self._available
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            self._available = None
            raise EmbeddingError("Ollama did not respond while checking embedding models.") from exc
        except httpx.HTTPError as exc:
            self._available = None
            raise EmbeddingError(
                f"Cannot reach the Ollama API at {self.base_url}."
            ) from exc
        except ValueError as exc:
            self._available = None
            raise EmbeddingProtocolError("Ollama returned invalid model-list JSON.") from exc

        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            self._available = None
            raise EmbeddingProtocolError("Ollama returned an invalid model list.")
        expected = (
            self.model
            if ":" in self.model.rsplit("/", 1)[-1]
            else f"{self.model}:latest"
        )
        self._available = any(
            isinstance(item, dict)
            and item.get("name") in {self.model, expected}
            for item in models
        )
        return self._available

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in small requests and validate consistent vector sizes."""
        results: list[list[float]] = []
        async for batch in self.embed_batches(texts):
            results.extend(batch)
        return results

    async def embed_batches(
        self, texts: Sequence[str]
    ) -> AsyncIterator[list[list[float]]]:
        """Yield validated batches so large imports do not retain all vectors."""
        if not texts:
            return
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding inputs must not be empty.")
        if not await self.is_available():
            raise EmbeddingModelMissing(
                f"Embedding model {self.model!r} is not installed. Install it "
                f"manually with: ollama pull {self.model}. LowP will not download it."
            )

        dimensions: int | None = None
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset : offset + self.batch_size]
            vectors: list[list[float]] = []
            try:
                response = await self._client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": list(batch), "truncate": False},
                )
                response.raise_for_status()
                payload = response.json()
            except httpx.TimeoutException as exc:
                logger.warning("Ollama embedding request timed out: %s", exc)
                raise EmbeddingError("The Ollama embedding request timed out.") from exc
            except httpx.HTTPStatusError as exc:
                body = exc.response.text.lower()
                if exc.response.status_code == 404 or (
                    "model" in body and "not found" in body
                ):
                    self._available = False
                    raise EmbeddingModelMissing(
                        f"Embedding model {self.model!r} is not available. "
                        f"Install it manually with: ollama pull {self.model}. "
                        "LowP will not download it."
                    ) from exc
                raise EmbeddingError(
                    "The Ollama embedding request failed."
                ) from exc
            except httpx.HTTPError as exc:
                logger.warning("Ollama embedding request failed: %s", exc)
                raise EmbeddingError("The Ollama embedding request failed.") from exc
            except ValueError as exc:
                raise EmbeddingProtocolError(
                    "Ollama returned invalid embedding JSON."
                ) from exc

            embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise EmbeddingProtocolError(
                    "Ollama returned a mismatched embedding batch."
                )
            for embedding in embeddings:
                if (
                    not isinstance(embedding, list)
                    or not embedding
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        for value in embedding
                    )
                ):
                    raise EmbeddingProtocolError(
                        "Ollama returned an invalid embedding vector."
                    )
                if dimensions is None:
                    dimensions = len(embedding)
                elif len(embedding) != dimensions:
                    raise EmbeddingProtocolError(
                        "Ollama returned vectors with inconsistent dimensions."
                    )
                vectors.append([float(value) for value in embedding])
            yield vectors
