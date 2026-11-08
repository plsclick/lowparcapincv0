"""Safe inspection and lifecycle operations for LowP-owned PEFT adapters."""

import json
import re
import shutil
from pathlib import Path
from typing import Any

from lowparllm.ai.lora.training import TrainingError
from lowparllm.core.paths import data_dir

_ADAPTER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class AdapterManager:
    """Manage metadata and files produced by LowP training jobs."""

    @property
    def directory(self) -> Path:
        return data_dir() / "training" / "adapters"

    def list(self) -> tuple[dict[str, Any], ...]:
        directory = self.directory
        if not directory.exists():
            return ()
        if (
            directory.is_symlink()
            or directory.parent.is_symlink()
            or not directory.is_dir()
        ):
            raise TrainingError("Refusing to inspect an unsafe adapter directory.")
        adapters = []
        for path in sorted(directory.iterdir()):
            if path.is_symlink() or path.parent.is_symlink() or not path.is_dir():
                continue
            metadata = path / "metadata.json"
            if metadata.is_file() and not metadata.is_symlink():
                adapters.append(self._read(path.name))
        return tuple(adapters)

    def info(self, name: str) -> dict[str, Any]:
        return self._read(name)

    def activate(self, name: str, selected_model: str) -> None:
        self._read(name)
        raise TrainingError(
            f"Adapter {name!r} targets a Hugging Face checkpoint, not Ollama model "
            f"{selected_model!r}. Ollama cannot safely activate this PEFT adapter "
            "without a separately validated conversion and matching base model; "
            "the adapter remains stored and the selected model is unchanged."
        )

    @staticmethod
    def deactivate() -> dict[str, object]:
        return {
            "active": False,
            "message": "No PEFT adapter is active in LowP's Ollama chat workflow.",
        }

    def delete(self, name: str, *, confirmed: bool = False) -> dict[str, str]:
        if not confirmed:
            raise TrainingError(
                f"Deleting adapter {name!r} requires explicit confirmation: "
                f"/adapters delete {name} confirm"
            )
        metadata = self._read(name)
        path = self.directory / name
        if path.is_symlink() or path.parent.resolve() != self.directory.resolve():
            raise TrainingError("Refusing to delete an adapter outside LowP storage.")
        shutil.rmtree(path)
        return {"name": name, "base_model_path": str(metadata.get("base_model_path", ""))}

    def _read(self, name: str) -> dict[str, Any]:
        if not _ADAPTER_NAME.fullmatch(name):
            raise TrainingError("Invalid adapter name.")
        path = self.directory / name
        if path.is_symlink() or not path.is_dir():
            raise TrainingError(f"Adapter {name!r} was not found.")
        metadata_path = path / "metadata.json"
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise TrainingError(f"Adapter {name!r} has no safe metadata file.")
        try:
            metadata: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingError(f"Could not read adapter metadata for {name!r}.") from exc
        if (
            not isinstance(metadata, dict)
            or metadata.get("name") != name
            or metadata.get("status") != "completed"
            or metadata.get("adapter_path") != str(path)
        ):
            raise TrainingError(f"Adapter {name!r} metadata is invalid or incomplete.")
        return metadata
