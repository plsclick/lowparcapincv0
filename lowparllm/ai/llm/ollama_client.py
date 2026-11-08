"""Asynchronous local Ollama API client with newline-delimited streaming."""

import json
import logging
import shutil
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from lowparllm.ai.llm.generation_config import GenerationConfig
from lowparllm.ai.llm.models import ChatMessage, GenerationMetrics, OllamaStatus, StreamEvent

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Base class for expected Ollama API failures."""


class OllamaUnavailable(OllamaError):
    """Ollama's local API could not be reached."""


class OllamaModelMissing(OllamaError):
    """The configured exact model tag is not available locally."""


class OllamaRequestError(OllamaError):
    """Ollama returned a request or model-loading error."""


class OllamaProtocolError(OllamaError):
    """Ollama returned malformed or incomplete streaming data."""


class OllamaClient:
    """Use Ollama's local HTTP API without blocking the asyncio event loop."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 600.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=5.0),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def status(self) -> OllamaStatus:
        installed = shutil.which("ollama") is not None
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models") if isinstance(payload, dict) else None
            if not isinstance(models, list):
                raise OllamaProtocolError("Ollama returned an invalid model list.")
        except httpx.TimeoutException as exc:
            logger.warning("Ollama status request timed out: %s", exc)
            return OllamaStatus(installed, False, False, None, error="Ollama did not respond in time.")
        except (httpx.HTTPError, ValueError, OllamaProtocolError) as exc:
            logger.info("Ollama API is unavailable at %s: %s", self.base_url, exc)
            return OllamaStatus(installed, False, False, None, error="Ollama API is unavailable.")

        model_data = next(
            (
                item
                for item in models
                if isinstance(item, dict) and item.get("name") == self.model
            ),
            None,
        )
        loaded: bool | None = None
        try:
            response = await self._client.get(f"{self.base_url}/api/ps")
            response.raise_for_status()
            running = response.json()
            running_models = running.get("models") if isinstance(running, dict) else None
            if isinstance(running_models, list):
                loaded = any(
                    isinstance(item, dict)
                    and self.model in {item.get("name"), item.get("model")}
                    for item in running_models
                )
        except (httpx.HTTPError, ValueError):
            logger.debug("Ollama running-model state is unavailable.")

        details = await self._model_details() if model_data is not None else None
        return OllamaStatus(
            installed=installed,
            connected=True,
            model_available=model_data is not None,
            model_loaded=loaded,
            model_size=(
                model_data.get("size")
                if isinstance(model_data, dict)
                and isinstance(model_data.get("size"), int)
                else None
            ),
            model_details=details,
        )

    async def _model_details(self) -> str | None:
        try:
            response = await self._client.post(
                f"{self.base_url}/api/show",
                json={"model": self.model},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        details = payload.get("details") if isinstance(payload, dict) else None
        if not isinstance(details, dict):
            return None
        parts = [
            f"{label}: {details[key]}"
            for key, label in (
                ("family", "family"),
                ("parameter_size", "parameters"),
                ("quantization_level", "quantization"),
            )
            if details.get(key)
        ]
        return ", ".join(parts) or None

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        generation: GenerationConfig,
    ) -> AsyncIterator[StreamEvent]:
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models") if isinstance(payload, dict) else None
            if not isinstance(models, list):
                raise OllamaProtocolError("Ollama returned an invalid model list.")
        except httpx.TimeoutException as exc:
            raise OllamaUnavailable("Ollama did not respond in time.") from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Check that Ollama is installed and its service is running."
            ) from exc
        except ValueError as exc:
            raise OllamaProtocolError("Ollama returned invalid model-list JSON.") from exc
        if not any(
            isinstance(item, dict) and item.get("name") == self.model for item in models
        ):
            raise OllamaModelMissing(
                f"Configured model {self.model!r} is not installed locally. "
                "Resolve this manually; LowParLLM never downloads or substitutes models."
            )

        request_body = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "think": False,
            "keep_alive": generation.keep_alive,
            "options": generation.as_options(),
        }
        got_done = False
        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=request_body,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    self._raise_response_error(response.status_code, body)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise OllamaProtocolError(
                            "Ollama returned malformed streaming data."
                        ) from exc
                    if not isinstance(chunk, dict):
                        raise OllamaProtocolError("Ollama returned an invalid response chunk.")
                    if chunk.get("error"):
                        self._raise_api_error(str(chunk["error"]))
                    message = chunk.get("message", {})
                    content = message.get("content", "") if isinstance(message, dict) else ""
                    if not isinstance(content, str):
                        raise OllamaProtocolError("Ollama returned invalid message content.")
                    if content:
                        yield StreamEvent(content=content)
                    if chunk.get("done") is True:
                        got_done = True
                        yield StreamEvent(
                            done=True,
                            metrics=GenerationMetrics.from_response(chunk),
                        )
                        break
        except httpx.TimeoutException as exc:
            logger.warning("Ollama chat timed out: %s", exc)
            raise OllamaUnavailable("The Ollama request timed out.") from exc
        except httpx.HTTPError as exc:
            logger.warning("Ollama streaming connection failed: %s", exc)
            raise OllamaUnavailable("The connection to Ollama was interrupted.") from exc
        if not got_done:
            raise OllamaProtocolError("Ollama ended the response before completion.")

    @staticmethod
    def _raise_response_error(status_code: int, body: bytes) -> None:
        try:
            payload: Any = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        message = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(message, str):
            OllamaClient._raise_api_error(message)
        if status_code == 404:
            raise OllamaModelMissing("The configured model was not found by Ollama.")
        raise OllamaRequestError(f"Ollama returned HTTP {status_code}.")

    @staticmethod
    def _raise_api_error(message: str) -> None:
        logger.warning("Ollama API error: %s", message)
        if "model" in message.lower() and any(
            phrase in message.lower() for phrase in ("not found", "does not exist")
        ):
            raise OllamaModelMissing(
                "The configured model is not installed locally. "
                "Resolve this manually; no model was downloaded."
            )
        raise OllamaRequestError(f"Ollama could not complete the request: {message}")
