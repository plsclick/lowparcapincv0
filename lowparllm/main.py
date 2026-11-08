"""Application entry point shared by ``python -m lowparllm`` and ``lowp``."""

import asyncio
import importlib.util
import os
from pathlib import Path
import sqlite3
import shutil
import subprocess
import sys

from lowparllm.core.config import CONFIG_FILE, load_config
from lowparllm.core.logging import configure_logging
from lowparllm.core.paths import cache_dir, config_dir, data_dir


def _check_health() -> int:
    """Run the non-interactive required-component health check."""
    try:
        load_config()
        from lowparllm.services.health import installation_health

        checks = installation_health(config_dir() / CONFIG_FILE)
    except (OSError, ValueError) as exc:
        print(f"Health check failed: {exc}", file=sys.stderr)
        return 1
    for check in checks:
        print(f"{check['status']}: {check['component']} - {check['detail']}")
    return 1 if any(check["status"] == "FAILED" for check in checks) else 0


def _run_doctor() -> int:
    """Run non-interactive installation and optional-service diagnostics."""
    try:
        config = load_config()
        from lowparllm.services.health import installation_health
        from lowparllm.services.system_info import gather_environment

        checks = list(installation_health(config_dir() / CONFIG_FILE))
        environment = gather_environment(
            Path.cwd(),
            ollama_url=config.ollama_url,
            model=config.base_model,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Doctor could not complete: {exc}", file=sys.stderr)
        return 1

    system = environment["system"]
    software = environment["software"]
    ollama = software["ollama"]
    checks.extend(
        [
            {
                "component": "Operating system",
                "status": "OK",
                "detail": (
                    f"{system['os']} {system['distribution_version']} "
                    f"({system['architecture']}); {system['terminal']}"
                ),
            },
            {
                "component": "Git",
                "status": "OK" if software["git_available"] else "Warning",
                "detail": "Available" if software["git_available"] else "Not installed; Git integration is optional.",
            },
        ]
    )
    shell_name = (
        ("pwsh" if shutil.which("pwsh") else "powershell")
        if os.name == "nt"
        else Path(os.environ.get("SHELL", "bash")).name
    )
    shell = shutil.which(shell_name)
    checks.append(
        {
            "component": "Shell",
            "status": "OK" if shell else "Warning",
            "detail": shell or f"{shell_name} was not found on PATH.",
        }
    )
    checks.append(
        {
            "component": "Ollama",
            "status": (
                "OK"
                if ollama["service_available"]
                else "Warning"
                if ollama["installed"]
                else "Not configured"
            ),
            "detail": (
                f"{config.ollama_url}; installed={ollama['installed']}; "
                f"service={ollama['service_available']}"
            ),
        }
    )
    checks.append(
        {
            "component": "Selected model",
            "status": (
                "OK"
                if ollama["base_model_installed"]
                else "Warning"
                if ollama["service_available"]
                else "Not configured"
            ),
            "detail": (
                config.base_model
                if ollama["base_model_installed"]
                else f"{config.base_model} is not available; LowP will not download it automatically."
            ),
        }
    )
    installed_models = {
        item.get("name")
        for item in ollama.get("models", ())
        if isinstance(item, dict)
    }
    embedding_model_available = config.rag.embedding_model in installed_models
    checks.append(
        {
            "component": "RAG embedding model",
            "status": (
                "OK"
                if embedding_model_available
                else "Warning"
                if config.rag.enabled and ollama["service_available"]
                else "Not configured"
            ),
            "detail": (
                config.rag.embedding_model
                if embedding_model_available
                else f"{config.rag.embedding_model} is not available; LowP will not download it automatically."
            ),
        }
    )

    if config.web.enabled:
        try:
            from lowparllm.ai.rag.rag_service import RagService
            from lowparllm.ai.research.research_service import ResearchService

            async def check_searxng() -> tuple[bool, str]:
                rag = RagService(config)
                research = ResearchService(config, rag)
                try:
                    return await research.provider.health_check()
                finally:
                    try:
                        await research.close()
                    finally:
                        await rag.close()

            web_available, web_detail = asyncio.run(check_searxng())
        except (OSError, RuntimeError, ValueError) as exc:
            web_available, web_detail = False, str(exc)
        checks.append(
            {
                "component": "SearXNG",
                "status": "OK" if web_available else "Warning",
                "detail": web_detail or config.web.searxng_base_url,
            }
        )
    else:
        checks.append(
            {
                "component": "SearXNG",
                "status": "Not configured",
                "detail": "Web research is disabled in configuration.",
            }
        )

    sqlite_path = config.rag.sqlite_path.expanduser()
    if sqlite_path.is_file() and not sqlite_path.is_symlink():
        try:
            with sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
                integrity = db.execute("PRAGMA quick_check").fetchone()
            sqlite_status = (
                "OK"
                if integrity and integrity[0] == "ok"
                else "Warning"
            )
            sqlite_detail = str(sqlite_path) if sqlite_status == "OK" else str(integrity)
        except sqlite3.Error as exc:
            sqlite_status, sqlite_detail = "Warning", str(exc)
    elif sqlite_path.is_symlink():
        sqlite_status, sqlite_detail = "Warning", "Refusing to inspect a symbolic-link database."
    else:
        sqlite_status, sqlite_detail = "Not configured", str(sqlite_path)
    checks.append(
        {
            "component": "Knowledge SQLite database",
            "status": sqlite_status,
            "detail": sqlite_detail,
        }
    )
    qdrant_path = config.rag.qdrant_path.expanduser()
    qdrant_available = qdrant_path.is_dir() and not qdrant_path.is_symlink()
    checks.append(
        {
            "component": "Local Qdrant store",
            "status": "OK" if qdrant_available else "Not configured",
            "detail": str(qdrant_path),
        }
    )
    try:
        from lowparllm.ai.lora.adapters import AdapterManager

        adapters = AdapterManager().list()
    except (OSError, RuntimeError, ValueError) as exc:
        adapter_status, adapter_detail = "Warning", str(exc)
    else:
        adapter_status = "OK"
        adapter_detail = (
            f"{len(adapters)} stored adapter(s); PEFT inference is unsupported by the current Ollama backend."
        )
    checks.append(
        {
            "component": "Stored adapters",
            "status": adapter_status,
            "detail": adapter_detail,
        }
    )
    training_modules = ("torch", "transformers", "peft", "accelerate")
    try:
        missing_training = tuple(
            name for name in training_modules if importlib.util.find_spec(name) is None
        )
    except (ImportError, ValueError):
        missing_training = training_modules
    checks.append(
        {
            "component": "Optional training dependencies",
            "status": "OK" if not missing_training else "Not configured",
            "detail": (
                "Training packages are installed."
                if not missing_training
                else "Not installed: " + ", ".join(missing_training)
            ),
        }
    )

    expected_launcher_directory = (
        Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        / "Programs"
        / "LowP"
        / "bin"
        if os.name == "nt"
        else Path.home() / ".local" / "bin"
    )
    path_entries = {
        os.path.normcase(os.path.normpath(item.strip().strip('"')))
        for item in os.environ.get("PATH", "").split(os.pathsep)
        if item.strip()
    }
    expected_path = os.path.normcase(
        os.path.normpath(str(expected_launcher_directory))
    )
    launcher = shutil.which("lowp")
    checks.append(
        {
            "component": "Launcher and PATH",
            "status": "OK" if launcher and expected_path in path_entries else "Warning",
            "detail": (
                launcher
                if launcher
                else f"Add {expected_launcher_directory} to the user PATH."
            ),
        }
    )
    for label, path in (
        ("Configuration directory", config_dir()),
        ("Application data directory", data_dir()),
        ("Application cache directory", cache_dir()),
    ):
        accessible = (
            path.is_dir()
            and not path.is_symlink()
            and os.access(path, os.R_OK | os.W_OK | os.X_OK)
        )
        checks.append(
            {
                "component": label,
                "status": "OK" if accessible else "Warning",
                "detail": str(path) if accessible else f"Not present or not accessible: {path}",
            }
        )

    for check in checks:
        print(f"{check['status']}: {check['component']} - {check['detail']}")
    return 1 if any(check["status"] == "FAILED" for check in checks) else 0


def _run_maintenance(action: str) -> int:
    """Delegate update/repair to the source checkout used for installation."""
    source_directory = os.environ.get("LOWP_SOURCE_DIR")
    installer_name = "install.ps1" if os.name == "nt" else "install.sh"
    installer = Path(source_directory).expanduser() / installer_name if source_directory else None
    if installer is None or not installer.is_file() or installer.is_symlink():
        command = (
            f".\\install.ps1 -{action.title()}"
            if os.name == "nt"
            else f"./install.sh --{action}"
        )
        print(
            f"Cannot run {action}: the installing source checkout is unavailable. "
            f"Run {command} from a LowP checkout.",
            file=sys.stderr,
        )
        return 1
    try:
        if os.name == "nt":
            powershell = shutil.which("pwsh") or shutil.which("powershell")
            if powershell is None:
                raise FileNotFoundError(
                    "PowerShell was not found. Install PowerShell or run install.ps1 manually."
                )
            switch = "Repair" if action == "repair" else "Update"
            command = [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(installer),
                f"-{switch}",
            ]
        else:
            bash = shutil.which("bash")
            if bash is None:
                raise FileNotFoundError("Bash was not found.")
            command = [bash, str(installer), f"--{action}"]
        return subprocess.run(
            command, check=False
        ).returncode
    except OSError as exc:
        print(f"Could not start the LowP {action} workflow: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    """Run the terminal interface or a non-interactive maintenance action."""
    arguments = sys.argv[1:]
    if arguments == ["--check"]:
        return _check_health()
    if arguments == ["doctor"]:
        return _run_doctor()
    if arguments in (["repair"], ["update"]):
        return _run_maintenance(arguments[0])
    if arguments:
        print("Usage: lowp [doctor|--check|repair|update]", file=sys.stderr)
        return 2
    try:
        config = load_config()
        configure_logging(config)
        from lowparllm.cli.terminal import TerminalApp

        return asyncio.run(TerminalApp(config).run())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except ModuleNotFoundError as exc:
        if exc.name in {
            "docx",
            "httpcore",
            "httpx",
            "prompt_toolkit",
            "pypdf",
            "qdrant_client",
            "rich",
        }:
            print(
                "LowParLLM terminal dependencies are missing. Install them with "
                "`./install.sh` on Linux or `.\\install.ps1` on Windows, then "
                "run `lowp repair` if needed.",
                file=sys.stderr,
            )
            return 1
        raise
    except ValueError as exc:
        print(f"Invalid LowParLLM configuration: {exc}", file=sys.stderr)
        return 2
