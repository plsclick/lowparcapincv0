"""Ollama model discovery, capability reporting, switching, and explicit pulls."""

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
import re
import tempfile
from typing import Any

import httpx

from lowparllm.ai.llm.ollama_client import OllamaClient
from lowparllm.core.config import AppConfig, CONFIG_FILE, validate_model_tag
from lowparllm.core.paths import config_dir


@dataclass(frozen=True)
class OllamaModel:
    name: str
    digest: str | None
    size_bytes: int | None
    modified_at: str | None
    family: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    model_format: str | None = None
    capabilities: tuple[str, ...] = ()
    context_length: int | None = None
    chat_compatible: bool | None = None
    training_compatible: bool = False


class ModelManagerError(RuntimeError):
    """A model-management operation could not be completed safely."""


def validate_model_name(name: str) -> str:
    return validate_model_tag(name)


def persist_model_name(name: str) -> None:
    """Atomically persist only [model].name without replacing other settings."""
    model_name = validate_model_name(name)
    path = config_dir() / CONFIG_FILE
    if path.is_symlink():
        raise ValueError(f"Refusing to edit a symbolic-link configuration file: {path}")
    try:
        original = path.read_text(encoding="utf-8")
        mode = path.stat().st_mode & 0o777
    except OSError as exc:
        raise ValueError(f"Could not read configuration file {path}: {exc}") from exc

    replacement = f"name = {json.dumps(model_name)}"
    lines = original.splitlines(keepends=True)
    section_start: int | None = None
    section_end = len(lines)
    for index, line in enumerate(lines):
        header = re.match(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$", line)
        if not header:
            continue
        if section_start is not None:
            section_end = index
            break
        if header.group(1).strip() == "model":
            section_start = index
    if section_start is None:
        updated = original
        if updated and not updated.endswith("\n"):
            updated += "\n"
        if updated and not updated.endswith("\n\n"):
            updated += "\n"
        updated += f"[model]\n{replacement}\n"
    else:
        key_line = next(
            (
                index
                for index in range(section_start + 1, section_end)
                if re.match(r"^\s*name\s*=", lines[index])
            ),
            None,
        )
        if key_line is None:
            lines.insert(section_start + 1, replacement + "\n")
        else:
            original_line = lines[key_line]
            comment = re.search(r"\s+#.*?(?=\r?\n?$)", original_line)
            ending = "\n" if original_line.endswith("\n") else ""
            comment_text = f" {comment.group(0).lstrip()}" if comment else ""
            lines[key_line] = replacement + comment_text + ending
        updated = "".join(lines)
    try:
        import tomllib

        tomllib.loads(updated)
    except (ImportError, ValueError) as exc:
        raise ValueError("Refusing to write invalid model configuration.") from exc
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".config.toml.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(updated)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, path)
    except OSError as exc:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise ValueError(f"Could not update model configuration in {path}: {exc}") from exc


class ModelManager:
    """Manage installed Ollama models while preserving the active conversation."""

    def __init__(
        self,
        config: AppConfig,
        client: OllamaClient,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(15, connect=5, read=None),
            follow_redirects=False,
            trust_env=False,
        )
        self._owns_http = http_client is None

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def list_models(self) -> tuple[OllamaModel, ...]:
        try:
            response = await self._http.get(f"{self.config.ollama_url}/api/tags")
            response.raise_for_status()
            payload: Any = response.json()
        except httpx.TimeoutException as exc:
            raise ModelManagerError("Ollama did not respond in time.") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelManagerError(
                f"Could not list models from Ollama at {self.config.ollama_url}."
            ) from exc
        raw_models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(raw_models, list):
            raise ModelManagerError("Ollama returned an invalid installed-model list.")
        if any(not isinstance(item, dict) for item in raw_models):
            raise ModelManagerError("Ollama returned invalid model metadata in its model list.")
        return tuple(_model_from_tags(item) for item in raw_models)

    async def model_info(self, name: str) -> OllamaModel:
        model_name = validate_model_name(name)
        models = await self.list_models()
        if model_name not in {model.name for model in models}:
            raise ModelManagerError(f"Model {model_name!r} is not installed in Ollama.")
        try:
            response = await self._http.post(
                f"{self.config.ollama_url}/api/show", json={"model": model_name}
            )
            response.raise_for_status()
            payload: Any = response.json()
        except httpx.TimeoutException as exc:
            raise ModelManagerError("Ollama model inspection timed out.") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelManagerError(f"Could not inspect model {model_name!r}.") from exc
        if not isinstance(payload, dict):
            raise ModelManagerError("Ollama returned invalid model metadata.")
        tag = next(model for model in models if model.name == model_name)
        model_info = payload.get("model_info")
        details = payload.get("details")
        capabilities = payload.get("capabilities")
        info = model_info if isinstance(model_info, dict) else {}
        detail = details if isinstance(details, dict) else {}
        capability_names = (
            tuple(item for item in capabilities if isinstance(item, str))
            if isinstance(capabilities, list)
            else ()
        )
        context = next(
            (
                value
                for key, value in info.items()
                if key.endswith(".context_length")
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
            ),
            None,
        )
        return OllamaModel(
            name=model_name,
            digest=tag.digest,
            size_bytes=tag.size_bytes,
            modified_at=tag.modified_at,
            family=detail.get("family") if isinstance(detail.get("family"), str) else None,
            parameter_size=(
                detail.get("parameter_size")
                if isinstance(detail.get("parameter_size"), str)
                else None
            ),
            quantization=(
                detail.get("quantization_level")
                if isinstance(detail.get("quantization_level"), str)
                else None
            ),
            model_format=detail.get("format") if isinstance(detail.get("format"), str) else None,
            capabilities=capability_names,
            context_length=context,
            chat_compatible=(
                True if "completion" in capability_names
                else False if capability_names
                else None
            ),
            training_compatible=False,
        )

    async def use(self, name: str) -> OllamaModel:
        model_name = validate_model_name(name)
        info = await self.model_info(model_name)
        if info.chat_compatible is False:
            raise ModelManagerError(
                f"Model {model_name!r} does not report Ollama chat-completion "
                "capability and was not selected."
            )
        persist_model_name(model_name)
        self.config.base_model = model_name
        self.client.model = model_name
        return info

    async def pull(
        self,
        name: str,
        *,
        confirmed: bool,
        progress: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        model_name = validate_model_name(name)
        if not confirmed:
            raise ModelManagerError(
                f"Model download needs explicit confirmation. Run "
                f"/model pull {model_name} confirm after reviewing the model."
            )
        try:
            async with self._http.stream(
                "POST",
                f"{self.config.ollama_url}/api/pull",
                json={"name": model_name, "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event: Any = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ModelManagerError(
                            "Ollama returned malformed model-download progress."
                        ) from exc
                    if not isinstance(event, dict):
                        raise ModelManagerError(
                            "Ollama returned invalid model-download progress."
                        )
                    if isinstance(event.get("error"), str):
                        raise ModelManagerError(
                            f"Ollama model download failed: {event['error']}"
                        )
                    if progress is not None:
                        progress(event)
        except httpx.TimeoutException as exc:
            raise ModelManagerError("Ollama model download timed out.") from exc
        except httpx.HTTPError as exc:
            raise ModelManagerError(
                "Could not download the model from Ollama. Check Ollama and network status."
            ) from exc
        if model_name not in {model.name for model in await self.list_models()}:
            raise ModelManagerError(
                "Ollama finished the pull stream, but the model is not listed as installed."
            )

    async def hardware_summary(self) -> dict[str, object]:
        from lowparllm.services.system_info import gather_environment

        report = await asyncio.to_thread(
            gather_environment,
            ollama_url=self.config.ollama_url,
            model=self.config.base_model,
            inspect_training_runtime=True,
        )
        return {
            "cpu": report["cpu"],
            "memory": report["memory"],
            "gpu": report["gpu"],
            "ollama": report["software"]["ollama"],
            "disk_free": report["software"]["project_disk_free"],
        }


def _model_from_tags(item: Mapping[str, Any]) -> OllamaModel:
    name = item.get("name")
    if not isinstance(name, str) or not name:
        raise ModelManagerError("Ollama returned a model without a name.")
    details = item.get("details")
    details = details if isinstance(details, dict) else {}
    size = item.get("size")
    return OllamaModel(
        name=name,
        digest=item.get("digest") if isinstance(item.get("digest"), str) else None,
        size_bytes=size if isinstance(size, int) and not isinstance(size, bool) else None,
        modified_at=(
            item.get("modified_at")
            if isinstance(item.get("modified_at"), str)
            else None
        ),
        family=details.get("family") if isinstance(details.get("family"), str) else None,
        parameter_size=(
            details.get("parameter_size")
            if isinstance(details.get("parameter_size"), str)
            else None
        ),
        quantization=(
            details.get("quantization_level")
            if isinstance(details.get("quantization_level"), str)
            else None
        ),
        model_format=details.get("format") if isinstance(details.get("format"), str) else None,
    )
