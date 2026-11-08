"""Read-only Linux, hardware, software, and local Ollama inspection."""

from __future__ import annotations

import ctypes
import json
import importlib
import os
from pathlib import Path
import platform
import re
import shutil
import shlex
import subprocess
import sys
from typing import Any, Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen

from lowparllm.core.config import CONVERSATIONAL_BASE_MODEL

BASE_MODEL = CONVERSATIONAL_BASE_MODEL
OLLAMA_URL = os.environ.get("LOWP_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_command(
    args: list[str], timeout: float = 4.0
) -> subprocess.CompletedProcess[str] | None:
    """Run an optional utility without a shell; return None if unavailable."""
    if shutil.which(args[0]) is None:
        return None
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def read_os_release() -> Mapping[str, str]:
    if platform.system() == "Windows":
        release = platform.release()
        version = platform.version()
        try:
            build = int(version.rsplit(".", 1)[-1])
        except ValueError:
            build = 0
        if release == "10" and build >= 22000:
            release = "11"
        return {
            "PRETTY_NAME": f"Windows {release}".strip(),
            "VERSION_ID": version,
        }
    try:
        return platform.freedesktop_os_release()
    except (AttributeError, OSError):
        values: dict[str, str] = {}
        try:
            for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator:
                    values[key] = value.strip().strip('"').strip("'")
        except OSError:
            pass
        return values


def cpu_information() -> tuple[str, int | None, int | None]:
    if platform.system() == "Windows":
        physical_cores = None
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell:
            value = _successful_output(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | "
                    "Measure-Object -Property NumberOfCores -Sum).Sum",
                ]
            )
            try:
                physical_cores = int(value) if value else None
            except ValueError:
                physical_cores = None
        return platform.processor() or "Unavailable", physical_cores, os.cpu_count()
    model = "Unavailable"
    cores: set[tuple[str, str]] = set()
    processor_count = 0
    physical_id = "0"
    core_id: str | None = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines() + [""]:
            key, separator, value = line.partition(":")
            if not separator:
                if core_id is not None:
                    cores.add((physical_id, core_id))
                physical_id, core_id = "0", None
                continue
            key, value = key.strip(), value.strip()
            if key in {"model name", "Hardware"} and model == "Unavailable":
                model = value
            elif key == "physical id":
                physical_id = value
            elif key == "core id":
                core_id = value
            elif key == "processor":
                processor_count += 1
    except OSError:
        pass
    logical_threads = os.cpu_count() or (processor_count or None)
    physical_cores = len(cores) or None
    return model, physical_cores, logical_threads


def memory_information() -> dict[str, int | None]:
    if platform.system() == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return {
                    "total": int(status.total_physical),
                    "available": int(status.available_physical),
                    "swap_total": None,
                    "swap_used": None,
                }
        except (AttributeError, OSError):
            pass
        return {
            "total": None,
            "available": None,
            "swap_total": None,
            "swap_used": None,
        }
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            match = re.match(r"^(\w+):\s+(\d+)\s+kB", line)
            if match:
                values[match.group(1)] = int(match.group(2)) * 1024
    except OSError:
        pass
    total = values.get("MemTotal")
    swap_total = values.get("SwapTotal")
    swap_free = values.get("SwapFree")
    return {
        "total": total,
        "available": values.get("MemAvailable"),
        "swap_total": swap_total,
        "swap_used": (
            max(0, swap_total - swap_free)
            if swap_total is not None and swap_free is not None
            else None
        ),
    }


def _successful_output(args: list[str]) -> str | None:
    result = run_command(args)
    if result is None or result.returncode != 0:
        return None
    return result.stdout.strip()


def gpu_information(*, inspect_training_runtime: bool = False) -> dict[str, Any]:
    gpu_output = _successful_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, str]] = []
    if gpu_output:
        for line in gpu_output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 3:
                gpus.append(
                    {
                        "name": fields[0],
                        "vendor": "NVIDIA",
                        "vram_mib": fields[1],
                        "driver": fields[2],
                    }
                )
    if not gpus:
        if platform.system() == "Windows":
            powershell = shutil.which("powershell") or shutil.which("pwsh")
            pci_output = (
                _successful_output(
                    [
                        powershell,
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        "Get-CimInstance Win32_VideoController | "
                        "ForEach-Object { $_.Name }",
                    ]
                )
                if powershell
                else None
            )
            if pci_output:
                gpus.extend(
                    {
                        "name": line.strip(),
                        "vendor": "Unavailable",
                        "vram_mib": "Unavailable",
                        "driver": "Unavailable",
                    }
                    for line in pci_output.splitlines()
                    if line.strip()
                )
        else:
            pci_output = _successful_output(["lspci", "-mm"])
        if pci_output and platform.system() != "Windows":
            for line in pci_output.splitlines():
                try:
                    fields = shlex.split(line)
                except ValueError:
                    continue
                if (
                    len(fields) >= 4
                    and fields[1] in {
                        "VGA compatible controller",
                        "3D controller",
                        "Display controller",
                    }
                ):
                    gpus.append(
                        {
                            "name": f"{fields[2]} {fields[3]}",
                            "vendor": fields[2],
                            "vram_mib": "Unavailable",
                            "driver": "Unavailable",
                        }
                    )

    cuda_output = _successful_output(["nvcc", "--version"])
    cuda_version = None
    if cuda_output:
        match = re.search(r"release\s+([\d.]+)", cuda_output)
        cuda_version = match.group(1) if match else "Detected (version unavailable)"
    cuda_runtime_available: bool | None = None
    runtime_cuda_version = None
    if inspect_training_runtime:
        try:
            torch = importlib.import_module("torch")
            cuda_runtime_available = bool(torch.cuda.is_available())
            runtime_cuda_version = getattr(torch.version, "cuda", None)
        except (ImportError, OSError, RuntimeError):
            cuda_runtime_available = None
    if runtime_cuda_version:
        cuda_version = str(runtime_cuda_version)

    return {
        "gpus": gpus,
        "nvidia_driver": shutil.which("nvidia-smi") is not None and any(
            item.get("vendor") == "NVIDIA" for item in gpus
        ),
        "cuda_available": (
            cuda_runtime_available
            if cuda_runtime_available is not None
            else bool(
                any(item.get("vendor") == "NVIDIA" for item in gpus)
                and cuda_output is not None
            )
        ),
        "cuda_runtime_available": cuda_runtime_available,
        "cuda_toolkit_available": cuda_output is not None,
        "cuda_version": cuda_version,
        "nvidia_utility_available": shutil.which("nvidia-smi") is not None,
    }


def _api_request(
    path: str,
    payload: dict[str, Any] | None = None,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{(base_url or OLLAMA_URL).rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="GET" if data is None else "POST",
    )
    try:
        with urlopen(request, timeout=2.0) as response:
            parsed = json.loads(response.read().decode("utf-8"))
            return parsed if isinstance(parsed, dict) else None
    except (OSError, URLError, TimeoutError, ValueError):
        return None


def ollama_information(
    base_url: str | None = None,
    model: str = BASE_MODEL,
) -> dict[str, Any]:
    ollama_installed = shutil.which("ollama") is not None
    version = _successful_output(["ollama", "--version"])
    list_result = run_command(["ollama", "list"], timeout=8.0)
    models: list[dict[str, str]] = []
    if list_result is not None and list_result.returncode == 0:
        for line in list_result.stdout.splitlines()[1:]:
            columns = line.split(maxsplit=3)
            if len(columns) >= 3:
                size = columns[2]
                modified = ""
                if len(columns) == 4:
                    size_match = re.match(r"^(\d+(?:\.\d+)?)\s+(\S+)(?:\s+(.*))?$", columns[2] + " " + columns[3])
                    if size_match:
                        size = f"{size_match.group(1)} {size_match.group(2)}"
                        modified = size_match.group(3) or ""
                models.append(
                    {
                        "name": columns[0],
                        "id": columns[1],
                        "size": size,
                        "modified": modified,
                    }
                )

    api_tags = _api_request("/api/tags", base_url=base_url)
    service_available = api_tags is not None
    if service_available and not models:
        for item in api_tags.get("models", []):
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                models.append(
                    {
                        "name": item["name"],
                        "id": str(item.get("digest", "")),
                        "size": format_bytes(item.get("size")),
                        "modified": "",
                    }
                )

    model_entry = next(
        (entry for entry in models if entry["name"] == model), None
    )
    details = (
        _api_request("/api/show", {"model": model}, base_url)
        if model_entry and service_available
        else None
    )
    detail_parts: list[str] = []
    if details:
        model_details = details.get("details", {})
        if isinstance(model_details, dict):
            for key in ("format", "family", "parameter_size", "quantization_level"):
                value = model_details.get(key)
                if value:
                    detail_parts.append(f"{key.replace('_', ' ')}: {value}")

    running_models = (
        _api_request("/api/ps", base_url=base_url) if service_available else None
    )
    loaded: bool | None = None
    if isinstance(running_models, dict) and isinstance(
        running_models.get("models"), list
    ):
        loaded = any(
            isinstance(item, dict)
            and model in {item.get("name"), item.get("model")}
            for item in running_models["models"]
        )

    return {
        "installed": ollama_installed,
        "version": version or "Unavailable",
        "service_available": service_available,
        "service_status": "Available" if service_available else "Unavailable",
        "models": models,
        "base_model_installed": model_entry is not None,
        "base_model": model_entry,
        "model_tag": model,
        "model_loaded": loaded,
        "model_details": ", ".join(detail_parts) if detail_parts else "Unavailable",
    }


def format_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "Unavailable"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return "Unavailable"


def _terminal_name() -> str:
    if platform.system() == "Windows":
        if os.environ.get("WT_SESSION"):
            return "Windows Terminal"
        terminal_program = os.environ.get("TERM_PROGRAM")
        if terminal_program:
            return terminal_program
        if os.environ.get("PSModulePath"):
            return "PowerShell"
        return "Windows console"
    for variable in ("KONSOLE_VERSION", "KONSOLE_DBUS_SERVICE"):
        if os.environ.get(variable):
            return "Konsole"
    terminal_program = os.environ.get("TERM_PROGRAM")
    if terminal_program:
        return terminal_program

    pid = os.getppid()
    for _ in range(5):
        try:
            process = Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
            if process.lower() in {"konsole", "gnome-terminal-server", "xterm"}:
                return {"konsole": "Konsole"}.get(process.lower(), process)
            status = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            fields = status[status.rfind(")") + 2 :].split()
            pid = int(fields[1])
            if pid <= 1:
                break
        except (OSError, ValueError, IndexError):
            break
    return "Unavailable"


def gather_environment(
    project_root: Path = PROJECT_ROOT,
    *,
    ollama_url: str | None = None,
    model: str = BASE_MODEL,
    inspect_training_runtime: bool = False,
) -> dict[str, Any]:
    release = read_os_release()
    desktop = os.environ.get("XDG_CURRENT_DESKTOP") or os.environ.get("DESKTOP_SESSION")
    desktop_name = "Windows desktop" if platform.system() == "Windows" else "Unavailable"
    if desktop:
        desktop_name = "KDE Plasma" if "kde" in desktop.lower() or "plasma" in desktop.lower() else desktop

    cpu_model, physical_cores, logical_threads = cpu_information()
    disk = shutil.disk_usage(project_root)
    pip_result = run_command([sys.executable, "-m", "pip", "--version"])
    pip_available = pip_result is not None and pip_result.returncode == 0
    git_available = shutil.which("git") is not None
    ollama = ollama_information(ollama_url, model)
    return {
        "system": {
            "os": platform.system() or "Unavailable",
            "distribution": release.get("PRETTY_NAME", release.get("NAME", "Unavailable")),
            "distribution_version": release.get("VERSION_ID", "Unavailable"),
            "kernel": platform.release() or "Unavailable",
            "architecture": platform.machine() or "Unavailable",
            "desktop": desktop_name,
            "terminal": _terminal_name(),
        },
        "cpu": {
            "model": cpu_model,
            "physical_cores": physical_cores,
            "logical_threads": logical_threads,
        },
        "memory": memory_information(),
        "gpu": gpu_information(
            inspect_training_runtime=inspect_training_runtime
        ),
        "software": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "pip_available": pip_available,
            "git_available": git_available,
            "ollama": ollama,
            "project_disk_free": disk.free,
        },
    }


def _display(value: Any) -> str:
    return "Unavailable" if value is None or value == "" else str(value)


def render_report(info: Mapping[str, Any]) -> str:
    system, cpu = info["system"], info["cpu"]
    memory, gpu, software = info["memory"], info["gpu"], info["software"]
    ollama = software["ollama"]
    model = ollama["base_model"]
    gpu_names = "; ".join(
        f'{entry["name"]} ({entry["vram_mib"]} MiB VRAM)' for entry in gpu["gpus"]
    ) or "Unavailable"
    lines = [
        "LOWPARLLM V0 — ENVIRONMENT REPORT",
        "",
        "SYSTEM",
        f'Operating System: {_display(system["os"])}',
        f'Distribution: {_display(system["distribution"])}',
        f'Distribution Version: {_display(system["distribution_version"])}',
        f'Kernel: {_display(system["kernel"])}',
        f'Architecture: {_display(system["architecture"])}',
        f'Desktop: {_display(system["desktop"])}',
        f'Terminal: {_display(system["terminal"])}',
        "",
        "CPU",
        f'Processor: {_display(cpu["model"])}',
        f'Physical Cores: {_display(cpu["physical_cores"])}',
        f'Logical Threads: {_display(cpu["logical_threads"])}',
        "",
        "MEMORY",
        f'Total RAM: {format_bytes(memory["total"])}',
        f'Available RAM: {format_bytes(memory["available"])}',
        f'Swap Capacity: {format_bytes(memory["swap_total"])}',
        f'Swap Used: {format_bytes(memory["swap_used"])}',
        "",
        "GPU",
        f"GPU: {gpu_names}",
        f'NVIDIA Driver: {"Available" if gpu["nvidia_driver"] else "Unavailable"}',
        "NVIDIA Utility: "
        + ("Available" if gpu["nvidia_utility_available"] else "Unavailable"),
        "CUDA Runtime: "
        + (
            f'Available ({gpu["cuda_version"]})'
            if gpu["cuda_available"]
            else "Unavailable"
        ),
        "CUDA Toolkit: "
        + ("Available" if gpu["cuda_toolkit_available"] else "Unavailable"),
        "",
        "SOFTWARE",
        f'Python: {software["python"]}',
        f'Python Executable: {software["python_executable"]}',
        f'pip: {"Available" if software["pip_available"] else "Unavailable"}',
        f'Git: {"Available" if software["git_available"] else "Unavailable"}',
        f'Ollama: {"Available" if ollama["installed"] else "Unavailable"}',
        f'Ollama Service: {ollama["service_status"]}',
        f'Project Filesystem Free: {format_bytes(software["project_disk_free"])}',
        "",
        "MODEL",
        f'Configured Model: {"Installed" if ollama["base_model_installed"] else "Missing"}',
        f'Model Tag: {ollama["model_tag"]}',
        f'Model Size: {_display(model["size"] if model else None)}',
        f'Model Details: {_display(ollama["model_details"])}',
        "",
        "STATUS",
        "Environment inspection completed.",
    ]
    return "\n".join(lines)


def main() -> int:
    try:
        print(render_report(gather_environment()))
    except OSError as exc:
        print(f"Environment inspection failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
