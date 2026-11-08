"""Local-only LoRA/QLoRA job configuration and process management."""

from dataclasses import asdict, dataclass
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any

from lowparllm.ai.lora.datasets import DatasetValidationError, get_dataset
from lowparllm.core.paths import data_dir

_JOB_ID = re.compile(r"^[0-9a-f]{12}$")
_TRAINING_DEPENDENCIES = ("torch", "transformers", "peft", "accelerate")


class TrainingError(RuntimeError):
    """A local model training operation could not be completed."""


@dataclass(frozen=True)
class TrainingConfig:
    base_model_path: str
    dataset_name: str
    method: str = "lora"
    epochs: float = 1.0
    learning_rate: float = 0.0002
    batch_size: int = 1
    gradient_accumulation: int = 8
    max_sequence_length: int = 1024
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    checkpoint_steps: int = 50
    output_directory: str | None = None

    def validate(self) -> "TrainingConfig":
        model_path = Path(self.base_model_path).expanduser()
        if model_path.is_symlink() or not model_path.is_dir():
            raise TrainingError(
                "Training requires a local Hugging Face model checkpoint directory; "
                "an Ollama model tag cannot be trained directly."
            )
        if (model_path / "config.json").is_symlink() or not (model_path / "config.json").is_file():
            raise TrainingError("Base checkpoint must contain a Hugging Face config.json.")
        if not any(
            (model_path / filename).is_file()
            and not (model_path / filename).is_symlink()
            for filename in (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
        ):
            raise TrainingError("Base checkpoint has no recognized local model weights.")
        if self.method not in {"lora", "qlora"}:
            raise TrainingError("Training method must be lora or qlora.")
        if not 0.1 <= self.epochs <= 100:
            raise TrainingError("Epochs must be between 0.1 and 100.")
        if not 1e-7 <= self.learning_rate <= 0.1:
            raise TrainingError("Learning rate must be between 0.0000001 and 0.1.")
        if not 1 <= self.batch_size <= 16:
            raise TrainingError("Batch size must be between 1 and 16.")
        if not 1 <= self.gradient_accumulation <= 128:
            raise TrainingError("Gradient accumulation must be between 1 and 128.")
        if not 128 <= self.max_sequence_length <= 8192:
            raise TrainingError("Sequence length must be between 128 and 8192.")
        if not 1 <= self.lora_rank <= 128:
            raise TrainingError("LoRA rank must be between 1 and 128.")
        if not 1 <= self.lora_alpha <= 256:
            raise TrainingError("LoRA alpha must be between 1 and 256.")
        if not 0 <= self.lora_dropout <= 0.5:
            raise TrainingError("LoRA dropout must be between 0 and 0.5.")
        if not 1 <= self.checkpoint_steps <= 10000:
            raise TrainingError("Checkpoint interval must be between 1 and 10000 steps.")
        if self.output_directory is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.output_directory
        ):
            raise TrainingError(
                "Output directory must be a simple name (letters, numbers, dots, "
                "underscores or hyphens)."
            )
        try:
            get_dataset(self.dataset_name)
        except DatasetValidationError as exc:
            raise TrainingError(str(exc)) from exc
        return self


class TrainingManager:
    """Launch and inspect isolated training worker processes."""

    def __init__(self) -> None:
        self.root = data_dir() / "training"
        self.jobs_dir = self.root / "jobs"
        self.adapters_dir = self.root / "adapters"
        self.outputs_dir = self.root / "outputs"

    @staticmethod
    def missing_dependencies() -> tuple[str, ...]:
        missing = []
        for package in _TRAINING_DEPENDENCIES:
            try:
                available = importlib.util.find_spec(package) is not None
            except (ImportError, ValueError):
                available = False
            if not available:
                missing.append(package)
        return tuple(missing)

    def start(self, config: TrainingConfig) -> dict[str, object]:
        config.validate()
        missing = self.missing_dependencies()
        if missing:
            raise TrainingError(
                "Optional training packages are not installed: "
                + ", ".join(missing)
                + '. Install the optional LowP training extra with '
                '`python -m pip install -e ".[training]"` and configure PyTorch '
                "for your hardware first."
            )
        self._prepare_directories()
        resource_check(config)
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(mode=0o700)
        config_path = job_dir / "job.json"
        status_path = job_dir / "status.json"
        log_path = job_dir / "training.log"
        spec = asdict(config)
        spec.update(
            {
                "job_id": job_id,
                "job_dir": str(job_dir),
                "adapter_dir": str(self.adapters_dir / job_id),
                "output_dir": str(
                    self.outputs_dir
                    / (config.output_directory or "jobs")
                    / job_id
                ),
                "resume": False,
            }
        )
        _write_json(config_path, spec)
        _write_json(
            status_path,
            {
                "job_id": job_id,
                "state": "starting",
                "started_at": time.time(),
                "config": asdict(config),
                "log_path": str(log_path),
            },
        )
        self._spawn(job_dir, log_path)
        return _read_json(status_path)

    def resume(self, job_id: str) -> dict[str, object]:
        job_dir = self._job_dir(job_id)
        status_path = job_dir / "status.json"
        status = _read_json(status_path)
        if status.get("state") not in {"failed", "cancelled", "interrupted"}:
            raise TrainingError("Only stopped or failed jobs can be resumed.")
        spec = _read_json(job_dir / "job.json")
        checkpoints = sorted(
            Path(str(spec["output_dir"])).glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1])
            if path.name.rsplit("-", 1)[-1].isdigit()
            else -1,
        )
        if not checkpoints:
            raise TrainingError("This job has no saved checkpoint to resume from.")
        config = status.get("config")
        if not isinstance(config, dict):
            raise TrainingError("Saved training configuration is invalid.")
        spec_path = job_dir / "job.json"
        spec["resume"] = True
        _write_json(spec_path, spec)
        status.update({"state": "starting", "resumed_at": time.time()})
        _write_json(status_path, status)
        self._spawn(job_dir, job_dir / "training.log")
        return _read_json(status_path)

    def status(self, job_id: str | None = None) -> tuple[dict[str, object], ...]:
        self._prepare_directories()
        selected = [self._job_dir(job_id)] if job_id else sorted(
            (path for path in self.jobs_dir.iterdir() if _JOB_ID.fullmatch(path.name)),
            key=lambda path: path.name,
            reverse=True,
        )
        result = []
        for path in selected:
            status_path = path / "status.json"
            if status_path.is_file() and not status_path.is_symlink():
                result.append(self._refresh_status(path, status_path))
        return tuple(result)

    def stop(self, job_id: str) -> dict[str, object]:
        job_dir = self._job_dir(job_id)
        status_path = job_dir / "status.json"
        status = _read_json(status_path)
        if status.get("state") not in {"starting", "running"}:
            raise TrainingError("That training job is not currently running.")
        pid = status.get("pid")
        if not isinstance(pid, int) or pid <= 1 or not _is_worker_process(pid, job_id):
            raise TrainingError(
                "The saved worker process could not be verified; refusing to signal it."
            )
        status.update({"state": "stopping", "stop_requested_at": time.time()})
        _write_json(status_path, status)
        try:
            if os.name == "nt":
                os.kill(pid, signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(pid, signal.SIGTERM)
        except OSError as exc:
            latest = _read_json(status_path)
            if latest.get("state") in {"cancelled", "completed", "failed"}:
                return latest
            raise TrainingError(f"Could not stop training worker {pid}.") from exc
        return _read_json(status_path)

    def tail_log(self, job_id: str, *, lines: int = 30) -> tuple[str, ...]:
        if not 1 <= lines <= 200:
            raise TrainingError("Log line count must be between 1 and 200.")
        path = self._job_dir(job_id) / "training.log"
        if not path.is_file() or path.is_symlink():
            return ()
        return tuple(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])

    def _prepare_directories(self) -> None:
        if self.root.is_symlink():
            raise TrainingError("Refusing to use a symbolic-link training directory.")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for directory in (self.jobs_dir, self.adapters_dir, self.outputs_dir):
            if directory.is_symlink() or directory.parent.is_symlink():
                raise TrainingError(f"Refusing to use symbolic-link path {directory}.")
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _job_dir(self, job_id: str) -> Path:
        if not _JOB_ID.fullmatch(job_id):
            raise TrainingError("Invalid training job identifier.")
        path = self.jobs_dir / job_id
        if path.is_symlink() or not path.is_dir():
            raise TrainingError(f"Training job {job_id} was not found.")
        return path

    @staticmethod
    def _refresh_status(job_dir: Path, status_path: Path) -> dict[str, Any]:
        status = _read_json(status_path)
        if status.get("state") in {"starting", "running", "stopping"}:
            pid = status.get("pid")
            job_id = job_dir.name
            if isinstance(pid, int) and not _is_worker_process(pid, job_id):
                status.update(
                    {
                        "state": "interrupted",
                        "error": (
                            "Training worker is no longer running. Resume is available "
                            "if a complete checkpoint was saved."
                        ),
                        "finished_at": time.time(),
                    }
                )
                _write_json(status_path, status)
        return status

    def _spawn(self, job_dir: Path, log_path: Path) -> None:
        spec_path = job_dir / "job.json"
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "lowparllm.ai.lora.worker", str(spec_path)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                close_fds=True,
                **(
                    {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                    if os.name == "nt"
                    else {"start_new_session": True}
                ),
            )
        status_path = job_dir / "status.json"
        status = _read_json(status_path)
        if status.get("state") in {"starting", "running"}:
            status.update({"state": "running", "pid": process.pid})
        _write_json(status_path, status)


def resource_check(config: TrainingConfig) -> dict[str, object]:
    """Apply conservative disk/RAM/VRAM checks using observed machine values."""
    model_path = Path(config.base_model_path).expanduser()
    model_bytes = sum(
        path.stat().st_size
        for pattern in ("*.safetensors", "pytorch_model*.bin")
        for path in model_path.glob(pattern)
        if path.is_file() and not path.is_symlink()
    )
    if model_bytes <= 0:
        raise TrainingError("Could not measure local checkpoint weight files.")
    disk_path = data_dir()
    while not disk_path.exists() and disk_path != disk_path.parent:
        disk_path = disk_path.parent
    available_disk = shutil.disk_usage(disk_path).free
    minimum_disk = model_bytes * 2 + 1024 * 1024 * 1024
    if available_disk < minimum_disk:
        raise TrainingError(
            "Insufficient disk space for checkpoints and adapter output: need at least "
            f"{minimum_disk // (1024 * 1024)} MiB free, have "
            f"{available_disk // (1024 * 1024)} MiB."
        )
    if config.method == "qlora":
        if shutil.which("nvidia-smi") is None:
            raise TrainingError(
                "QLoRA is supported only with a detected NVIDIA CUDA GPU in this workflow."
            )
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise TrainingError("QLoRA requires the optional PyTorch training package.") from exc
        if not torch.cuda.is_available() or torch.version.cuda is None:
            raise TrainingError(
                "The installed PyTorch package has no working NVIDIA CUDA runtime."
            )
        free_mib = _nvidia_free_memory_mib()
        if max(free_mib) < 6144:
            raise TrainingError(
                "QLoRA requires at least 6 GiB free GPU memory in this supported "
                "configuration; less is currently available."
            )
        if importlib.util.find_spec("bitsandbytes") is None:
            raise TrainingError(
                "QLoRA requires the optional bitsandbytes CUDA package. Install and "
                "verify it for the installed driver and PyTorch build first."
            )
    else:
        try:
            torch = importlib.import_module("torch")
        except (ImportError, OSError):
            torch = None
        if torch is not None and torch.cuda.is_available():
            if shutil.which("nvidia-smi") is None or torch.version.cuda is None:
                raise TrainingError(
                    "This LoRA workflow can preflight active GPU training only for "
                    "NVIDIA CUDA; use a supported setup or free enough system RAM."
                )
            free_mib = _nvidia_free_memory_mib()
            minimum_vram = max(4096, int(model_bytes * 1.5 / (1024 * 1024)))
            if max(free_mib) < minimum_vram:
                raise TrainingError(
                    "The detected GPU has less free VRAM than the conservative LoRA "
                    f"preflight estimate of {minimum_vram} MiB."
                )
        else:
            try:
                from lowparllm.services.system_info import memory_information

                available_ram = memory_information().get("available")
            except (ImportError, OSError):
                available_ram = None
            if isinstance(available_ram, int) and available_ram < model_bytes * 3:
                raise TrainingError(
                    "Available system RAM is below the conservative 3× checkpoint-size "
                    "LoRA preflight estimate. Reduce the model size or free memory; "
                    "this estimate is not a guarantee of successful training."
                )
    return {
        "checkpoint_bytes": model_bytes,
        "minimum_disk_bytes": minimum_disk,
        "available_disk_bytes": available_disk,
    }


def _nvidia_free_memory_mib() -> list[int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TrainingError("Could not query free NVIDIA GPU memory.") from exc
    free_mib = [
        int(value.strip())
        for value in result.stdout.splitlines()
        if value.strip().isdigit()
    ]
    if result.returncode or not free_mib:
        raise TrainingError("Could not determine currently free NVIDIA GPU memory.")
    return free_mib


def _write_json(path: Path, value: dict[str, object]) -> None:
    if path.is_symlink():
        raise TrainingError(f"Refusing to write through symbolic link {path}.")
    descriptor, temporary = tempfile.mkstemp(prefix=".lowp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TrainingError(f"Training metadata is missing or unsafe: {path}.")
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"Could not read training metadata from {path}.") from exc
    if not isinstance(value, dict):
        raise TrainingError(f"Training metadata in {path} is invalid.")
    return value


def _is_worker_process(pid: int, job_id: str) -> bool:
    if os.name == "nt":
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            return False
        command = (
            f"$process = Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}'; "
            "if ($process -and $process.CommandLine -and "
            "$process.CommandLine.Contains('lowparllm.ai.lora.worker') -and "
            f"$process.CommandLine.Contains('{job_id}')) {{ exit 0 }} else {{ exit 1 }}"
        )
        try:
            result = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return b"lowparllm.ai.lora.worker" in command and job_id.encode() in command
