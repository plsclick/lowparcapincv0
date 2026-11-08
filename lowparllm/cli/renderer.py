"""Rich-based terminal rendering, kept separate from command behavior."""

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lowparllm.ai.coding.agent import AgentOutcome
from lowparllm.ai.coding.command_runner import CommandResult as DevelopmentCommandResult
from lowparllm.ai.coding.workspace import FileSnapshot, ProjectInfo
from lowparllm.cli.command_registry import command_help
from lowparllm.ai.llm.models import GenerationMetrics
from lowparllm.ai.llm.model_manager import OllamaModel
from lowparllm.ai.rag.citations import Citation
from lowparllm.ai.rag.retriever import RetrievedChunk
from lowparllm.ai.rag.rag_service import IngestionResult
from lowparllm.ai.research.citations import WebCitation
from lowparllm.ai.search.models import SearchResult, WebSource
from lowparllm.core.config import AppConfig
from lowparllm.storage.database import SourceRecord
from lowparllm.services.system_info import format_bytes

_URL = re.compile(r"(?<![=(<])https?://[^\s<>]+", re.IGNORECASE)
_FENCED_CODE = re.compile(
    r"(?ms)^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n"
    r"(?P<code>.*?)^[ \t]*(?P=fence)[ \t]*$"
)


def _value(value: Any) -> str:
    return "Unavailable" if value is None or value == "" else str(value)


def _linkify_plain_urls(markdown: str) -> str:
    parts: list[str] = []
    cursor = 0
    for match in _FENCED_CODE.finditer(markdown):
        parts.append(_linkify_text(markdown[cursor:match.start()]))
        parts.append(match.group())
        cursor = match.end()
    parts.append(_linkify_text(markdown[cursor:]))
    return "".join(parts)


def _linkify_text(text: str) -> str:
    def replace_url(match: re.Match[str]) -> str:
        url = match.group()
        suffix = ""
        while url and url[-1] in ".,;:!?":
            suffix = url[-1] + suffix
            url = url[:-1]
        while url.endswith(")") and url.count("(") < url.count(")"):
            suffix = ")" + suffix
            url = url[:-1]
        return f"<{url}>{suffix}" if url else match.group()

    return _URL.sub(replace_url, text)


class Renderer:
    """Render application status and conversation content in the terminal."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._stream_buffer: list[str] = []
        self._last_code_blocks: tuple[str, ...] = ()
        self._clipboard_process: subprocess.Popen[bytes] | None = None
        self._live_response: Live | None = None

    def show_startup(self, config: AppConfig, status: dict[str, Any]) -> None:
        system = status["system"]
        cpu = status["cpu"]
        memory = status["memory"]
        software = status["software"]
        gpu = status["gpu"]
        ollama = software["ollama"]
        gpu_name = gpu["gpus"][0]["name"] if gpu["gpus"] else "Unavailable"
        available_memory = format_bytes(memory["available"])
        ollama_status = "Available" if ollama["service_available"] else "Unavailable"
        model_status = "Installed" if ollama["base_model_installed"] else "Missing"

        body = Table.grid(padding=(0, 2))
        body.add_column(style="bold cyan")
        body.add_column()
        body.add_row("System", _value(system["distribution"]))
        body.add_row("CPU", _value(cpu["model"]))
        body.add_row("Available RAM", available_memory)
        body.add_row("GPU", gpu_name)
        body.add_row("Ollama", ollama_status)
        body.add_row(config.base_model, model_status)
        self.console.print(
            Panel(
                body,
                title=f"[bold]Welcome to {config.app_name}[/bold]",
                subtitle=f"Version {config.app_version}",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )
        self.console.print(
            "[dim]Local Ollama chat · session-only conversation · persistent local knowledge[/dim]"
        )
        self.console.print(
            "[dim]Enter submits slash commands; Enter adds message lines; "
            "Esc+Enter submits a message. Type /help for commands.[/dim]\n"
        )

    def show_workspace(self, path: str) -> None:
        self.console.print(f"[dim]Working directory: {path}[/dim]")

    def show_help(self, commands: tuple[str, ...]) -> None:
        grouped: dict[str, list[str]] = {}
        for command in commands:
            grouped.setdefault(command_help(command).category, []).append(command)
        width = max(20, self.console.size.width)
        max_name_length = max((len(command) for command in commands), default=8)
        max_description_length = max(
            (len(command_help(command).description) for command in commands),
            default=32,
        )
        min_column_width = max(
            34,
            min(78, max_name_length + min(44, max_description_length) + 3),
        )
        available_columns = max(1, width // min_column_width)
        desired_columns = max(2, (len(commands) + 11) // 12)
        column_count = min(
            available_columns,
            desired_columns,
            max(1, len(commands)),
        )
        if width < 100:
            column_count = 1
        self.console.print("[bold]Available commands[/bold]")
        for category, category_commands in grouped.items():
            columns = min(column_count, len(category_commands))
            rows = (len(category_commands) + columns - 1) // columns
            table = Table(
                title=category,
                box=box.SIMPLE,
                show_header=False,
                expand=True,
                padding=(0, 1),
            )
            for _ in range(columns):
                table.add_column(
                    ratio=1,
                    overflow="fold",
                    no_wrap=False,
                    min_width=min(18, width // columns),
                )
            for row in range(rows):
                cells = []
                for column in range(columns):
                    index = column * rows + row
                    if index >= len(category_commands):
                        cells.append("")
                        continue
                    command = category_commands[index]
                    metadata = command_help(command)
                    cells.append(
                        Text.assemble(
                            (command, "cyan"),
                            (" — ", "dim"),
                            (metadata.description, ""),
                        )
                    )
                table.add_row(*cells)
            self.console.print(table)
        self.console.print(
            "[dim]Assistant response links are clickable. Use /copy <number> "
            "to copy a generated code block.[/dim]"
        )

    def show_route_decision(self, plan: Any, model: str) -> None:
        table = Table(title="LowP task router", box=box.SIMPLE)
        table.add_column("Decision", style="cyan")
        table.add_column("Result")
        table.add_row("Intent", plan.intent.value.replace("_", " ").title())
        table.add_row("Complexity", plan.complexity)
        table.add_row("Confidence", f"{plan.confidence:.0%}")
        table.add_row("Base model", model)
        table.add_row("Adapter", "None (no compatible adapter inference backend)")
        table.add_row(
            "Information source",
            plan.research_mode or "None; direct local-model response",
        )
        table.add_row(
            "Capabilities",
            ", ".join(plan.required_capabilities) or "None",
        )
        self.console.print(table)
        for index, step in enumerate(plan.steps, 1):
            self.console.print(f"  {index}. {step}")
        for note in plan.notes:
            self.console.print(f"[yellow]Note: {note}[/yellow]")

    def show_router_status(self, payload: dict[str, Any]) -> None:
        config = payload["config"]
        table = Table(title="Routing preferences", box=box.SIMPLE)
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        table.add_row("Automatic routing", "On" if config.automatic else "Off")
        table.add_row(
            "Automatic model selection",
            "On" if config.automatic_model_selection else "Off",
        )
        table.add_row(
            "Automatic adapter selection",
            "On" if config.automatic_adapter_selection else "Off",
        )
        table.add_row("Configured model", payload["default_model"])
        table.add_row("Adapter inference", "Unsupported for current Ollama/PEFT setup")
        if payload.get("last_route") is not None:
            table.add_row(
                "Last intent",
                payload["last_route"].intent.value.replace("_", " ").title(),
            )
        self.console.print(table)
        self.console.print(
            "[dim]Commands: /router plan <request>, /router explain, "
            "/router capabilities, /router auto on|off[/dim]"
        )

    def show_doctor(self, payload: dict[str, Any]) -> None:
        table = Table(title="LowP health check", box=box.SIMPLE)
        table.add_column("Component", style="cyan")
        table.add_column("Status")
        table.add_column("Details", overflow="fold")
        for check in payload.get("checks", ()):
            status = str(check.get("status", "Unknown"))
            style = "green" if status in {"OK", "Available"} else "yellow"
            if status in {"FAILED", "Failed"}:
                style = "red"
            table.add_row(
                str(check.get("component", "Unknown")),
                f"[{style}]{status}[/{style}]",
                str(check.get("detail", "")),
            )
        self.console.print(table)
        failures = int(payload.get("required_failures", 0))
        self.console.print(
            "[red]Required installation checks failed.[/red]"
            if failures
            else "[dim]Required installation checks passed; optional services may be unavailable.[/dim]"
        )

    def show_pdf(self, action: str, payload: Any) -> None:
        if action == "pdf_inspection":
            table = Table(title="PDF information", box=box.SIMPLE)
            table.add_column("Property", style="cyan")
            table.add_column("Value")
            for key, value in payload.items():
                table.add_row(str(key), str(value))
            self.console.print(table)
        elif action in {"pdf_text", "pdf_summary"}:
            self.console.print(Panel(Markdown(str(payload)), title="PDF content"))
        elif action == "pdf_text_saved":
            self.console.print(f"[green]Saved extracted text:[/green] {payload}")
        else:
            self.console.print(Panel(str(payload), title="PDF reading"))

    def show_router_capabilities(self, capabilities: tuple[Any, ...]) -> None:
        table = Table(title="LowP capability inventory", box=box.SIMPLE)
        table.add_column("Capability", style="cyan")
        table.add_column("Status")
        table.add_column("Details")
        for capability in capabilities:
            table.add_row(
                capability.name,
                capability.status.value,
                capability.detail or capability.description,
            )
        self.console.print(table)

    def show_status(
        self,
        details: dict[str, Any],
        app_version: str,
    ) -> None:
        status = details["status"]
        system = status["system"]
        cpu = status["cpu"]
        memory = status["memory"]
        software = status["software"]
        gpu = status["gpu"]
        ollama = software["ollama"]
        table = Table(title=f"System status · LowParLLM {app_version}", box=box.SIMPLE)
        table.add_column("Component", style="cyan")
        table.add_column("Status")
        table.add_row(
            "Distribution",
            f'{_value(system["distribution"])} (kernel {_value(system["kernel"])})',
        )
        table.add_row("CPU", _value(cpu["model"]))
        table.add_row(
            "Memory",
            f'{format_bytes(memory["available"])} available / '
            f'{format_bytes(memory["total"])} total',
        )
        gpu_label = (
            ", ".join(
                f'{item["name"]} ({item["vram_mib"]} MiB VRAM)'
                for item in gpu["gpus"]
            )
            or "Unavailable"
        )
        table.add_row("GPU", gpu_label)
        table.add_row("Python", _value(software["python"]))
        table.add_row(
            "Ollama",
            f'{"Installed" if ollama["installed"] else "Unavailable"}; '
            f'API {"Connected" if ollama["service_available"] else "Unavailable"}',
        )
        table.add_row("Ollama endpoint", details["ollama_url"])
        table.add_row("Configured model", details["model"])
        table.add_row(
            "Local model",
            "Installed" if ollama["base_model_installed"] else "Missing",
        )
        loaded = ollama.get("model_loaded")
        table.add_row(
            "Model loaded",
            "Yes" if loaded is True else "No" if loaded is False else "Unknown",
        )
        table.add_row("Conversation messages", str(details["message_count"]))
        self.console.print(table)
        if not ollama["installed"]:
            self.console.print(
                "[yellow]Install/start Ollama manually to enable local inference "
                "when it is available.[/yellow]"
            )
        elif not ollama["service_available"]:
            self.console.print(
                "[yellow]Ollama is installed but its local service is unavailable. "
                "Start it manually when needed.[/yellow]"
            )
        if ollama["installed"] and not ollama["base_model_installed"]:
            self.console.print(
                "[yellow]The configured model is not listed locally. Check `ollama list`; "
                "LowParLLM will not download models.[/yellow]"
            )

    def show_model(self, config: AppConfig, status: dict[str, Any]) -> None:
        ollama = status["software"]["ollama"]
        table = Table(title="Conversational model", box=box.SIMPLE)
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        table.add_row("Configured model", config.base_model)
        table.add_row("Ollama endpoint", config.ollama_url)
        table.add_row(
            "Ollama installation",
            "Available" if ollama["installed"] else "Unavailable",
        )
        table.add_row("Ollama service", ollama["service_status"])
        table.add_row(
            "Exact model tag",
            "Installed" if ollama["base_model_installed"] else "Missing",
        )
        model = ollama["base_model"]
        table.add_row("Reported size", _value(model["size"] if model else None))
        table.add_row("Model details", _value(ollama["model_details"]))
        self.console.print(table)

    def show_models(self, active: str, models: tuple[OllamaModel, ...]) -> None:
        table = Table(title="Installed Ollama models", box=box.SIMPLE)
        table.add_column("Model", style="cyan")
        table.add_column("Active")
        table.add_column("Chat compatibility")
        table.add_column("Family")
        table.add_column("Parameters")
        table.add_column("Quantization")
        table.add_column("Size")
        for model in models:
            table.add_row(
                model.name,
                "Yes" if model.name == active else "",
                (
                    "Yes"
                    if model.chat_compatible is True
                    else "No"
                    if model.chat_compatible is False
                    else "Inspect with /model info"
                ),
                _value(model.family),
                _value(model.parameter_size),
                _value(model.quantization),
                format_bytes(model.size_bytes),
            )
        self.console.print(table)
        self.console.print(
            f"[dim]Selected model: {active}. Use /model use <name>, /model info <name>, "
            "/model refresh, or /model pull <name>.[/dim]"
        )
        self.console.print(
            "[yellow]Model size is not a memory-fit guarantee. Compare it with "
            "available RAM/VRAM in /status and allow extra runtime/context overhead.[/yellow]"
        )
        if not models:
            self.console.print("[yellow]Ollama returned no installed models.[/yellow]")

    def show_model_overview(self, payload: dict[str, Any]) -> None:
        active = payload["active"]
        selected = payload.get("selected_info")
        self.console.print(
            f"[bold]Selected default:[/bold] {active} "
            f"({'installed' if selected is not None else 'not installed'})"
        )
        environment = payload.get("environment")
        if isinstance(environment, dict):
            software = environment.get("software")
            ollama = software.get("ollama") if isinstance(software, dict) else None
            if isinstance(ollama, dict) and isinstance(
                ollama.get("model_loaded"), bool
            ):
                loaded = "loaded in memory" if ollama["model_loaded"] else "not loaded"
                self.console.print(f"[dim]Runtime status: {loaded}[/dim]")
        if selected is not None:
            compatibility = (
                "Chat-compatible"
                if selected.chat_compatible is True
                else "Not reported as chat-compatible"
                if selected.chat_compatible is False
                else "Chat compatibility not reported"
            )
            self.console.print(
                f"[dim]{compatibility}; context length: "
                f"{_value(selected.context_length)}; direct training: no "
                "(Ollama artifacts are not training checkpoints).[/dim]"
            )
            memory = environment.get("memory") if isinstance(environment, dict) else None
            gpu = environment.get("gpu") if isinstance(environment, dict) else None
            available_ram = memory.get("available") if isinstance(memory, dict) else None
            gpus = gpu.get("gpus") if isinstance(gpu, dict) else None
            if (
                isinstance(selected.size_bytes, int)
                and isinstance(available_ram, int)
                and isinstance(gpus, list)
            ):
                vram_values: list[int] = []
                for item in gpus:
                    if not isinstance(item, dict):
                        continue
                    try:
                        vram_values.append(int(item["vram_mib"]) * 1024 * 1024)
                    except (KeyError, TypeError, ValueError):
                        vram_values = []
                        break
                if len(vram_values) == len(gpus):
                    available_total = available_ram + sum(vram_values)
                    if selected.size_bytes > available_total:
                        self.console.print(
                            "[yellow]Warning: the installed model size exceeds the "
                            "currently available RAM plus reported GPU VRAM. It may "
                            "not fit; model size also excludes runtime/context overhead.[/yellow]"
                        )
        self.show_models(active, payload["models"])
        self.console.print(
            "[dim]Actions: /model list, /model refresh, /model info <name>, "
            "/model use <name>, /model pull <name> (then confirm explicitly).[/dim]"
        )

    def show_model_info(self, model: OllamaModel) -> None:
        table = Table(title=f"Model details · {model.name}", box=box.SIMPLE)
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        for label, value in (
            ("Digest", model.digest),
            ("Family", model.family),
            ("Parameters", model.parameter_size),
            ("Quantization", model.quantization),
            ("Format", model.model_format),
            ("Installed size", format_bytes(model.size_bytes)),
            ("Modified", model.modified_at),
            ("Context length", model.context_length),
            ("Capabilities", ", ".join(model.capabilities) or None),
            (
                "Chat compatible",
                "Yes"
                if model.chat_compatible is True
                else "No"
                if model.chat_compatible is False
                else "Not reported by Ollama",
            ),
            ("Direct training compatible", "No (Ollama models are inference artifacts)"),
        ):
            table.add_row(label, _value(value))
        self.console.print(table)
        self.console.print(
            "[dim]Training requires a separately obtained compatible Hugging Face "
            "checkpoint; no checkpoint is downloaded automatically.[/dim]"
        )

    def show_model_switched(self, name: str) -> None:
        self.console.print(
            f"[green]Default model changed to {name}. Existing conversation history was preserved.[/green]"
        )

    def show_model_pull_confirmation(self, name: str) -> None:
        self.console.print(
            f"[yellow]Pulling {name!r} may download a large model and use substantial "
            "disk space. Nothing has been downloaded.[/yellow]\n"
            f"Review its source and available disk space, then confirm with:\n"
            f"  /model pull {name} confirm"
        )

    def show_model_pull_progress(self, event: dict[str, object] | Any) -> None:
        status = event.get("status", "Downloading")
        completed = event.get("completed")
        total = event.get("total")
        if (
            isinstance(completed, int)
            and isinstance(total, int)
            and total > 0
        ):
            percent = max(0.0, min(100.0, completed / total * 100))
            self.console.print(
                f"[dim]Ollama {status}: {percent:.1f}% "
                f"({format_bytes(completed)} / {format_bytes(total)})[/dim]"
            )
        else:
            self.console.print(f"[dim]Ollama {status}[/dim]")

    def show_model_pulled(self, name: str) -> None:
        self.console.print(
            f"[green]Model {name} is installed. Select it with /model use {name}.[/green]"
        )

    def show_training(self, action: str, payload: Any) -> None:
        if action == "training_help":
            self.console.print(payload)
        elif action == "training_hardware":
            report = payload["report"]
            cpu = report["cpu"]
            memory = report["memory"]
            gpu = report["gpu"]
            table = Table(title="Training hardware preflight facts", box=box.SIMPLE)
            table.add_column("Resource", style="cyan")
            table.add_column("Detected value")
            table.add_row("CPU", str(cpu.get("model", "Unavailable")))
            table.add_row("System RAM total", format_bytes(memory.get("total")))
            table.add_row("System RAM available", format_bytes(memory.get("available")))
            table.add_row(
                "GPUs",
                ", ".join(
                    f"{item.get('name', 'Unknown')} · {item.get('vram_mib', '?')} MiB"
                    for item in gpu.get("gpus", ())
                )
                or "None detected",
            )
            table.add_row("NVIDIA driver", "Detected" if gpu.get("nvidia_driver") else "Not detected")
            runtime = gpu.get("cuda_runtime_available")
            table.add_row(
                "PyTorch CUDA runtime",
                "Available" if runtime is True else "Unavailable" if runtime is False else "Not checked",
            )
            table.add_row("Detected CUDA version", str(gpu.get("cuda_version") or "Unavailable"))
            table.add_row(
                "Free disk for LowP data",
                format_bytes(payload["training_data_disk_free"]),
            )
            ollama = report["software"]["ollama"]
            table.add_row(
                "Ollama/service",
                "Available" if ollama.get("service_available") else "Unavailable",
            )
            table.add_row(
                "Selected model",
                f"{ollama.get('model_tag', 'Unknown')} "
                f"({'installed' if ollama.get('base_model_installed') else 'missing'})",
            )
            table.add_row(
                "Training packages missing",
                ", ".join(payload["missing_dependencies"]) or "None",
            )
            self.console.print(table)
            self.console.print(
                "[dim]These detections do not guarantee that a particular model or "
                "training configuration will fit. QLoRA is currently limited to "
                "NVIDIA CUDA with at least 6 GiB free VRAM.[/dim]"
            )
        elif action == "training_dataset_list":
            table = Table(title="Imported training datasets", box=box.SIMPLE)
            table.add_column("Name", style="cyan")
            table.add_column("Format")
            table.add_column("Records", justify="right")
            table.add_column("SHA-256")
            for dataset in payload:
                table.add_row(
                    dataset.name,
                    dataset.format,
                    str(dataset.records),
                    dataset.digest[:16],
                )
            self.console.print(table)
            if not payload:
                self.console.print("[yellow]No explicitly imported datasets.[/yellow]")
        elif action == "training_dataset_validation":
            self._show_dataset_report(payload)
        elif action == "training_dataset_imported":
            self.console.print(
                f"[green]Imported {payload.name}: {payload.records} records "
                f"({payload.path}).[/green]"
            )
        elif action == "training_dataset_info":
            self.console.print(
                f"[bold cyan]{payload['dataset'].name}[/bold cyan] · "
                f"{payload['dataset'].path}"
            )
            self._show_dataset_report(payload["report"])
        elif action in {"training_started", "training_resumed", "training_stopped"}:
            self._show_training_jobs((payload,))
        elif action == "training_status":
            self._show_training_jobs(payload)
        elif action == "training_logs":
            self.console.print(f"[bold]Training log · {payload['job_id']}[/bold]")
            for line in payload["lines"]:
                self.console.print(line)
            if not payload["lines"]:
                self.console.print("[dim]No log output yet.[/dim]")
        elif action == "adapter_list":
            table = Table(title="Stored PEFT adapters", box=box.SIMPLE)
            table.add_column("Name", style="cyan")
            table.add_column("Method")
            table.add_column("Dataset")
            table.add_column("Base checkpoint")
            for adapter in payload:
                table.add_row(
                    str(adapter.get("name", "Unknown")),
                    str(adapter.get("training_method", "Unknown")),
                    str(adapter.get("dataset_name", "Unknown")),
                    str(adapter.get("base_model_path", "Unknown")),
                )
            self.console.print(table)
            if not payload:
                self.console.print("[yellow]No completed LowP adapters found.[/yellow]")
        elif action == "adapter_info":
            table = Table(title=f"Adapter · {payload.get('name', 'Unknown')}", box=box.SIMPLE)
            table.add_column("Field", style="cyan")
            table.add_column("Value")
            for key in (
                "status", "training_method", "base_model_path", "dataset_name",
                "trained_at", "adapter_path", "compatibility",
            ):
                table.add_row(key.replace("_", " ").title(), str(payload.get(key, "Unavailable")))
            self.console.print(table)
        elif action == "adapter_delete_confirmation":
            self.console.print(
                f"[yellow]Adapter files are user data and will be permanently deleted. "
                f"Confirm with `/adapters delete {payload} confirm`.[/yellow]"
            )
        elif action == "adapter_deleted":
            self.console.print(f"[green]Adapter {payload['name']} was deleted.[/green]")
        elif action == "adapter_deactivated":
            self.console.print(payload["message"])
        elif action == "adapter_activated":
            self.console.print(f"[green]Adapter {payload} activated.[/green]")

    def _show_dataset_report(self, report: Any) -> None:
        table = Table(title="Dataset validation", box=box.SIMPLE)
        table.add_column("Property", style="cyan")
        table.add_column("Value")
        for label, value in (
            ("Path", report.path),
            ("Format", report.format),
            ("Records", report.records),
            ("Valid unique records", report.valid_records),
            ("Duplicates", report.duplicates),
            ("Errors", len(report.errors)),
        ):
            table.add_row(label, str(value))
        self.console.print(table)
        for error in report.errors[:20]:
            self.console.print(f"[red]• {error}[/red]")
        if len(report.errors) > 20:
            self.console.print(f"[dim]… and {len(report.errors) - 20} more[/dim]")

    def _show_training_jobs(self, jobs: Any) -> None:
        table = Table(title="Training jobs", box=box.SIMPLE)
        table.add_column("Job", style="cyan")
        table.add_column("State")
        table.add_column("Progress")
        table.add_column("Method")
        table.add_column("Dataset")
        table.add_column("Last checkpoint")
        for job in jobs:
            config = job.get("config", {})
            step, maximum = job.get("step"), job.get("max_steps")
            progress = f"{step}/{maximum}" if step is not None else "Not started"
            table.add_row(
                str(job.get("job_id", "Unknown")),
                str(job.get("state", "Unknown")),
                progress,
                str(config.get("method", "Unknown")),
                str(config.get("dataset_name", "Unknown")),
                str(job.get("checkpoint", "—")),
            )
            if job.get("error"):
                self.console.print(f"[red]{job['error']}[/red]")
        self.console.print(table)
        if not jobs:
            self.console.print("[yellow]No training jobs found.[/yellow]")

    def show_settings(self, details: dict[str, Any]) -> None:
        config = details["config"]
        web_status = details.get("web_status") or {}
        table = Table(title="Current settings", box=box.SIMPLE)
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        for label, value in (
            ("Application", f"{config.app_name} {config.app_version}"),
            ("Ollama URL", config.ollama_url),
            ("Base model", config.base_model),
            ("Keep alive", config.keep_alive),
            ("Context size (num_ctx)", config.context_size),
            ("Output token limit (num_predict)", config.num_predict),
            ("Maximum history messages", config.max_history_messages),
            ("Temperature", config.temperature),
            ("Top P", config.top_p),
            ("Top K", config.top_k),
            ("Log level", config.log_level),
            (
                "Internet research",
                "On" if web_status.get("enabled", config.web.enabled) else "Off",
            ),
            (
                "Automatic web research",
                "On"
                if web_status.get("automatic_research", config.web.automatic_research)
                else "Off",
            ),
            ("SearXNG instance", config.web.searxng_base_url),
            ("Maximum search queries", config.web.max_queries),
            ("Maximum results per query", config.web.max_results),
            ("Maximum webpages", config.web.max_pages),
            ("Web request timeout", f"{config.web.timeout_seconds:g} seconds"),
            ("Maximum concurrent web requests", config.web.max_concurrent_requests),
            ("Maximum webpage text", f"{config.web.max_page_chars} characters"),
            (
                "Unlimited research quotas",
                "On"
                if web_status.get("unlimited", config.web.unlimited)
                else "Off",
            ),
            (
                "Allow saving web sources",
                "On"
                if web_status.get(
                    "save_sources_enabled", config.web.save_sources_enabled
                )
                else "Off",
            ),
            ("System prompt", config.system_prompt),
            ("Configuration directory", details["config_dir"]),
            ("Data directory", details["data_dir"]),
            ("Cache directory", details["cache_dir"]),
        ):
            table.add_row(label, str(value))
        self.console.print(table)

    def show_generation_settings(self, config: AppConfig) -> None:
        table = Table(title="Generation settings", box=box.SIMPLE)
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        table.add_row("Temperature", str(config.temperature))
        table.add_row("Top-p", str(config.top_p))
        table.add_row("Top-k", str(config.top_k))
        table.add_row("Keep-alive", config.keep_alive)
        self.console.print(table)
        self.console.print(
            "[dim]Change one with /generation <setting> <value>. "
            "Values are saved in the per-user LowP configuration.[/dim]"
        )

    def show_code_status(self, payload: dict[str, Any]) -> None:
        project = payload.get("project")
        self.console.print(
            f"Coding mode: {'on' if payload['active'] else 'off'} · "
            f"workspace: [cyan]{payload['workspace']}[/cyan]"
        )
        if project is not None:
            self.console.print(
                f"Project: {', '.join(project.languages) or 'unclassified'} · "
                f"Git: {project.git_branch or ('repository' if project.is_git_repository else 'not detected')}"
            )
        if payload.get("permission"):
            self.console.print(f"Workspace permission: {payload['permission']}")

    def show_code_help(self) -> None:
        self.console.print(
            "Coding mode: /code on|off|status|plan <task>|continue [confirm]|stop\n"
            "Workspace: /pwd, /ls [-a] [-l], /cd <path>, /tree [path], /find <pattern>\n"
            "Project: /project [analyze|structure|status|instructions|refresh|open <path>|index]\n"
            "Files: /file read <path>, /file create <path> [--confirm] <text>, "
            "/file write <path> <sha256> [--confirm] <text>, /file diff\n"
            "Git/development: /git status|diff|log|branch|review, /test, /build, /lint, /run <command>\n"
            "Search/review/debug: /search text|file|symbol <query>, /review [changes|file <path>|security|performance], /debug <error>\n"
            "Permissions: /permissions status|read-only|workspace-write|ask\n"
            "Approvals: inspect proposed changes first; approve agent edits with /code continue confirm "
            "and commands with /run confirm."
        )

    def show_code_stopped(self, stopped: bool) -> None:
        self.console.print(
            "Coding task stopped." if stopped else "No active coding task to stop."
        )

    def show_permissions(self, payload: dict[str, Any]) -> None:
        self.console.print(
            f"Permission mode: [bold]{payload['mode']}[/bold] · "
            f"workspace boundary: {payload['workspace']}"
        )
        self.console.print(
            "[dim]Path checks are application-level restrictions, not an OS sandbox. "
            "Commands run only after separate approval and without a shell.[/dim]"
        )

    def show_path(self, path: str) -> None:
        self.console.print(path)

    def show_directory_list(self, payload: dict[str, Any]) -> None:
        self.console.print(f"[bold]{payload['path']}[/bold]")
        for entry in payload["entries"]:
            self.console.print(entry)
        if not payload["entries"]:
            self.console.print("[dim](empty)[/dim]")

    def show_directory_changed(self, payload: dict[str, Any]) -> None:
        self.console.print(
            f"Working directory: [cyan]{payload['current']}[/cyan]"
        )

    def show_directory_tree(self, payload: dict[str, Any]) -> None:
        self.console.print("\n".join(payload["entries"]))

    def show_file_search(self, paths: tuple[str, ...]) -> None:
        for path in paths:
            self.console.print(path)
        if not paths:
            self.console.print("[dim]No matching files.[/dim]")

    def show_project_search(self, payload: dict[str, Any]) -> None:
        self.console.print(
            f"[bold]Project {payload['mode']} search:[/bold] {payload['query']}"
        )
        for match in payload["matches"]:
            self.console.print(match)
        if not payload["matches"]:
            self.console.print("[dim]No matching lines.[/dim]")

    def show_project_info(self, project: ProjectInfo) -> None:
        table = Table(title="Active project", box=box.SIMPLE)
        table.add_column("Property", style="cyan")
        table.add_column("Value")
        table.add_row("Directory", str(project.root))
        table.add_row("Git repository", "Yes" if project.is_git_repository else "No")
        table.add_row("Branch", project.git_branch or "—")
        table.add_row("Languages", ", ".join(project.languages) or "Not detected")
        table.add_row("Manifests", ", ".join(project.manifests) or "None detected")
        table.add_row(
            "Instructions",
            ", ".join(str(path) for path in project.instructions) or "None found",
        )
        self.console.print(table)
        if project.recommended_commands:
            self.console.print(
                "Detected checks: "
                + "; ".join(
                    f"{name}: {' '.join(argv)}"
                    for name, argv in project.recommended_commands
                )
            )

    def show_project_instructions(self, snapshots: tuple[FileSnapshot, ...]) -> None:
        for snapshot in snapshots:
            self.console.print(
                Panel(snapshot.content, title=str(snapshot.path), border_style="yellow")
            )
        if not snapshots:
            self.console.print("[dim]No project AGENTS.md instructions were found.[/dim]")

    def show_project_index_confirmation(self, payload: dict[str, Any]) -> None:
        self.console.print(
            f"Index supported documents under {payload['path']} into LowP's existing "
            "local knowledge base? Hidden and generated directories are excluded. "
            "Confirm with [cyan]/project index confirm[/cyan]."
        )

    def show_file_snapshot(self, snapshot: FileSnapshot) -> None:
        self.console.print(
            Panel(snapshot.content, title=str(snapshot.path), border_style="cyan")
        )
        self.console.print(f"[dim]SHA-256: {snapshot.sha256}[/dim]")

    def show_file_written(self, snapshot: FileSnapshot, diff: str = "") -> None:
        self.console.print(f"Updated [cyan]{snapshot.path}[/cyan].")
        self.show_file_diff(diff)

    def show_file_deleted(self, path: str) -> None:
        self.console.print(f"Deleted [cyan]{path}[/cyan].")

    def show_file_diff(self, diff: str) -> None:
        if diff:
            self.console.print(Panel(diff, title="File diff", border_style="yellow"))
        else:
            self.console.print("[dim]No file diff is available.[/dim]")

    def show_git_output(self, payload: dict[str, str]) -> None:
        self.console.print(Panel(
            payload["output"] or "(no changes)",
            title=f"Git {payload['action']}",
            border_style="cyan",
        ))

    def show_run_confirmation(self, payload: dict[str, Any]) -> None:
        command = shlex.join(payload["argv"])
        self.console.print(
            f"Command: [bold]{command}[/bold]\n"
            f"Working directory: {payload['cwd']}\n"
            "It will run without a shell. Review project scripts and side effects first. "
            "Approve this exact pending command with [cyan]/run confirm[/cyan]."
        )

    def show_command_result(self, result: DevelopmentCommandResult) -> None:
        title = (
            f"Command exit code {result.exit_code}"
            + (" · timed out" if result.timed_out else "")
        )
        self.console.print(Panel(
            f"$ {shlex.join(result.argv)}\n"
            f"cwd: {result.cwd}\n\n"
            f"stdout:\n{result.stdout or '(empty)'}\n\n"
            f"stderr:\n{result.stderr or '(empty)'}"
            + ("\n[output truncated]" if result.output_truncated else ""),
            title=title,
            border_style="green" if result.exit_code == 0 else "red",
        ))

    def show_agent_outcome(self, outcome: AgentOutcome) -> None:
        if outcome.message:
            self.console.print(Panel(outcome.message, title="LowP coding agent"))
        for change in outcome.changes:
            if change.get("written"):
                self.console.print(f"Updated [cyan]{change.get('path')}[/cyan].")
                if change.get("diff"):
                    self.show_file_diff(str(change["diff"]))
            elif change.get("deleted"):
                self.show_file_deleted(str(change.get("path")))
        if outcome.pending_approval:
            self.console.print("[bold yellow]Proposed operations — not yet executed:[/bold yellow]")
            for index, action in enumerate(outcome.actions, 1):
                self.console.print(f"{index}. {action.description()}")
                preview = action.arguments.get("preview")
                if preview:
                    self.console.print(Panel(str(preview), border_style="yellow"))
            self.console.print(
                "Inspect the proposed diffs and command above. Approve with "
                "[cyan]/code continue confirm[/cyan], or reject with [cyan]/code stop[/cyan]."
            )
        elif outcome.task_complete:
            self.console.print("[dim]Task turn complete.[/dim]")

    def show_init_confirmation(self, payload: dict[str, Any]) -> None:
        project: ProjectInfo = payload["project"]
        self.console.print(
            f"Create project-specific guidance at {payload['path']} for "
            f"{', '.join(project.languages) or 'this project'}? Existing files are "
            "never overwritten. Confirm with [cyan]/init confirm[/cyan]."
        )

    def show_user_message(self, message: str) -> None:
        self.console.print(Panel(Text(message), title="You", border_style="blue"))

    def show_assistant_message(self, message: str) -> None:
        self._render_response(message, title="LowParLLM")

    def begin_stream(self) -> None:
        self._stream_buffer = []
        self._live_response = Live(
            Markdown(""),
            console=self.console,
            refresh_per_second=12,
            transient=True,
        )
        self._live_response.start()

    def stream_token(self, content: str) -> None:
        self._stream_buffer.append(content)
        if self._live_response is not None:
            response = _linkify_plain_urls("".join(self._stream_buffer))
            self._live_response.update(Markdown(response))

    def finish_stream(self, metrics: GenerationMetrics | None) -> None:
        self._stop_live_response()
        self._render_response("".join(self._stream_buffer), title="LowP")
        self._stream_buffer.clear()
        details: list[str] = []
        if metrics is not None:
            if metrics.eval_count is not None:
                details.append(f'{metrics.eval_count} generated tokens')
            if metrics.prompt_eval_duration_ns is not None:
                details.append(
                    f"{metrics.prompt_eval_duration_ns / 1_000_000_000:.2f}s prompt eval"
                )
            if metrics.eval_duration_ns is not None:
                details.append(
                    f"{metrics.eval_duration_ns / 1_000_000_000:.2f}s generation"
                )
            rate = metrics.tokens_per_second
            if rate is not None:
                details.append(f"{rate:.1f} tokens/s")
            if metrics.total_duration_ns is not None:
                details.append(
                    f"{metrics.total_duration_ns / 1_000_000_000:.2f}s total"
                )
            if metrics.prompt_eval_count is not None:
                details.append(f'{metrics.prompt_eval_count} prompt tokens')
        summary = " · ".join(details)
        self.console.print(f"[dim]✓ complete{f' · {summary}' if summary else ''}[/dim]")

    def show_generation_stopped(self) -> None:
        self._stop_live_response()
        if self._stream_buffer:
            self._render_response("".join(self._stream_buffer), title="LowP (partial)")
            self._stream_buffer.clear()
        self.console.print("[yellow]Generation stopped.[/yellow]")

    def end_stream(self) -> None:
        self._stop_live_response()
        self._stream_buffer.clear()

    def _stop_live_response(self) -> None:
        if self._live_response is not None:
            self._live_response.stop()
            self._live_response = None

    def _render_response(self, response: str, *, title: str) -> None:
        self._last_code_blocks = tuple(
            match.group("code") for match in _FENCED_CODE.finditer(response)
        )
        rendered = _linkify_plain_urls(response)
        self.console.print(
            Panel(Markdown(rendered), title=title, border_style="green")
        )
        if self._last_code_blocks:
            self.console.print(
                "[dim]Copy a code block with /copy <number>, or select it and press "
                "Ctrl+Shift+C.[/dim]"
            )

    def copy_code_block(self, number: int) -> None:
        if not 1 <= number <= len(self._last_code_blocks):
            raise ValueError(
                f"No code block {number} in the latest response. "
                "Use /copy to see the available block numbers."
            )
        content = self._last_code_blocks[number - 1]
        data = content.encode("utf-8")
        if os.name == "nt":
            clip = shutil.which("clip")
            if clip is None:
                raise RuntimeError("Windows clipboard utility clip.exe was not found.")
            try:
                result = subprocess.run(
                    [clip],
                    input=data,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=5,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("clip.exe did not respond within 5 seconds.") from exc
            if result.returncode:
                message = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(message or "Could not copy to the system clipboard.")
            return

        wl_copy = shutil.which("wl-copy")
        if wl_copy:
            try:
                result = subprocess.run(
                    [wl_copy],
                    input=data,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=5,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("wl-copy did not respond within 5 seconds.") from exc
            if result.returncode:
                message = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(message or "Could not copy to the system clipboard.")
            return

        xsel = shutil.which("xsel")
        if xsel:
            try:
                result = subprocess.run(
                    [xsel, "--clipboard", "--input"],
                    input=data,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=5,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("xsel did not respond within 5 seconds.") from exc
            if result.returncode:
                message = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(message or "Could not copy to the system clipboard.")
            return

        xclip = shutil.which("xclip")
        if xclip:
            if self._clipboard_process is not None:
                self._clipboard_process.terminate()
            process = subprocess.Popen(
                [xclip, "-selection", "clipboard", "-in"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                if process.stdin is None:
                    raise RuntimeError("Could not open clipboard input.")
                process.stdin.write(data)
                process.stdin.close()
            except OSError as exc:
                process.terminate()
                raise RuntimeError("Could not copy to the system clipboard.") from exc
            if process.poll() is not None and process.returncode:
                error = process.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    error.strip() or "Could not copy to the system clipboard."
                )
            self._clipboard_process = process
            return

        raise RuntimeError(
            "No clipboard utility found. Install wl-clipboard, xsel, or xclip, "
            "or select the code and press Ctrl+Shift+C."
        )

    def show_copy_usage(self) -> None:
        if not self._last_code_blocks:
            self.console.print("[yellow]The latest response has no code blocks.[/yellow]")
            return
        self.console.print(
            "[dim]Available code blocks: "
            + ", ".join(str(index) for index in range(1, len(self._last_code_blocks) + 1))
            + ". Use /copy <number>.[/dim]"
        )

    def show_copy_success(self, number: int) -> None:
        self.console.print(f"[green]Copied code block {number} to clipboard.[/green]")

    def show_reset(self) -> None:
        self.console.print("[dim]In-memory conversation context reset.[/dim]")

    def show_context_pruned(self, message_count: int) -> None:
        if message_count:
            self.console.print(
                f"[dim]Older context trimmed to fit the configured context budget "
                f"({message_count} messages omitted from this turn).[/dim]"
            )

    def show_rag_searching(self) -> None:
        self.console.print("[dim]Searching local knowledge…[/dim]")

    def show_rag_unavailable(self, message: str) -> None:
        self.console.print(
            f"[yellow]Knowledge retrieval unavailable; continuing normal chat. "
            f"{message}[/yellow]"
        )

    def show_citations(
        self, citations: tuple[Citation | WebCitation, ...]
    ) -> None:
        if not citations:
            return
        self.console.print("\n[bold cyan]Sources[/bold cyan]")
        for citation in citations:
            self.console.print(Markdown(_linkify_text(citation.reference)))

    def show_web_status(self, status: dict[str, Any]) -> None:
        table = Table(title="SearXNG internet research", box=box.SIMPLE)
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        for label, key in (
            ("Enabled", "enabled"),
            ("Automatic web research", "automatic_research"),
            ("Unlimited research quotas", "unlimited"),
            ("SearXNG URL", "searxng_base_url"),
            ("Connection", "connection_status"),
            ("Default language", "default_language"),
            ("Search categories", "search_categories"),
            ("Safe search (0-2)", "safe_search"),
            ("Maximum queries", "max_queries"),
            ("Results per query", "max_results"),
            ("Maximum webpages", "max_pages"),
            ("Request timeout", "timeout_seconds"),
            ("Concurrent requests", "max_concurrent_requests"),
            ("Maximum response size (bytes)", "max_page_bytes"),
            ("Maximum webpage text", "max_page_chars"),
            ("Maximum research context text", "max_context_chars"),
            ("Save sources into knowledge", "save_sources_enabled"),
            ("Search cache TTL (seconds)", "search_cache_ttl_seconds"),
        ):
            value = status.get(key, False if key == "unlimited" else "Unavailable")
            if status.get("unlimited", False):
                value = {
                    "max_queries": "Up to 10 queries (hard safety cap)",
                    "max_results": "Up to 100 per query (hard cap)",
                    "max_pages": "Up to 25 pages (hard cap)",
                    "max_page_bytes": status["hard_max_page_bytes"],
                    "max_page_chars": "Bounded by response-size ceiling",
                    "max_context_chars": status["model_context_char_limit"],
                }.get(key, value)
            if isinstance(value, bool):
                value = "On" if value else "Off"
            table.add_row(label, str(value))
        if status.get("unlimited", False):
            table.add_row(
                "Per-page response hard ceiling",
                f'{status["hard_max_page_bytes"]} bytes',
            )
        self.console.print(table)
        if status.get("connection_error"):
            self.console.print(f"[yellow]{status['connection_error']}[/yellow]")
        self.console.print(
            "[dim]Change limits with /web-settings <setting> <value> "
            "(base_url, language, categories, safe_search, max_queries, "
            "max_results, max_pages, timeout_seconds, max_concurrent_requests, "
            "max_page_bytes, max_page_chars, max_context_chars). Toggle all "
            "per-task quotas with "
            "/web-settings unlimited on|off. Timeout, concurrency, SearXNG "
            "pagination, response-size, and model-context safety bounds remain. "
            "Use /web-cache-settings ttl <seconds> to configure search-cache expiry. "
            "Automatic web research is opt-in.[/dim]"
        )

    def show_web_cache(self, stats: dict[str, Any]) -> None:
        table = Table(title="SearXNG search cache", box=box.SIMPLE)
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        for label, key in (
            ("Entries", "entries"),
            ("Database size", "database_bytes"),
            ("Expires after (seconds)", "ttl_seconds"),
            ("Oldest entry", "oldest_entry"),
            ("Newest entry", "newest_entry"),
            ("Cache file", "path"),
        ):
            table.add_row(label, str(stats[key] if stats[key] is not None else "None"))
        self.console.print(table)
        self.console.print(
            "[dim]Use /web-cache-clear to remove cached results or "
            "/web-cache-settings ttl <seconds> to change expiration.[/dim]"
        )

    def show_web_cache_cleared(self, removed: int) -> None:
        self.console.print(f"[green]Cleared {removed} cached search result sets.[/green]")

    def show_web_search_results(self, results: tuple[SearchResult, ...]) -> None:
        if not results:
            self.console.print("[dim]The search provider returned no results.[/dim]")
            return
        self.console.print("[bold]Internet search results[/bold]")
        for index, result in enumerate(results, start=1):
            title = result.title or "Untitled result"
            detail = f"{result.domain}"
            if result.published_date:
                detail += f" · Published: {result.published_date}"
            content = Table.grid(padding=(0, 1))
            content.add_column()
            title_text = Text(f"[{index}] {title}", style="bold cyan")
            title_text.stylize(f"link {result.url}")
            content.add_row(title_text)
            content.add_row(Text(detail, style="dim"))
            content.add_row(Text(result.snippet[:600]))
            url_text = Text(result.url, style="blue")
            url_text.stylize(f"link {result.url}")
            content.add_row(url_text)
            self.console.print(Panel(content, border_style="blue"))
        self.console.print(
            "[dim]Use /web-open <URL> to inspect a page or /web-save <number> "
            "to save a result into local knowledge.[/dim]"
        )

    def show_web_page(self, source: WebSource) -> None:
        heading = Text(f"{source.title}\n{source.domain} · ")
        url_start = len(heading)
        heading.append(source.url)
        heading.stylize(f"link {source.url}", url_start, len(heading))
        if source.final_url and source.final_url != source.url:
            heading.append("\nRetrieved as: ")
            final_url_start = len(heading)
            heading.append(source.final_url)
            heading.stylize(
                f"link {source.final_url}", final_url_start, len(heading)
            )
        if source.published_date:
            heading.append(f"\nPublished: {source.published_date}")
        self.console.print(
            Panel(
                Markdown(_linkify_plain_urls(source.content)),
                title=heading,
                border_style="blue",
            )
        )

    def show_web_stage(self, stage: str) -> None:
        self.console.print(f"[dim]Web research: {stage}…[/dim]")

    def show_research_stage(self, mode: str, stage: str) -> None:
        labels = {
            "local": "Local research",
            "web": "Web research",
            "both": "Combined research",
        }
        label = labels.get(mode, "Research")
        self.console.print(f"[dim]{label}: {stage}…[/dim]")

    def show_web_unavailable(self, message: str) -> None:
        self.console.print(
            f"[yellow]Web research unavailable; continuing with local chat. "
            f"{message}[/yellow]"
        )

    def show_research_no_evidence(self, mode: str) -> None:
        self.console.print(
            f"[yellow]No usable sources were found for {mode} research; "
            "LowP will not invent citations.[/yellow]"
        )

    def show_research_errors(self, errors: tuple[str, ...]) -> None:
        for error in errors:
            self.console.print(f"[yellow]Research note: {error}[/yellow]")

    def show_web_saved(self, result: IngestionResult) -> None:
        self.show_ingestion_result(result)

    def show_knowledge(self, status: dict[str, Any]) -> None:
        table = Table(title="Local knowledge base", box=box.SIMPLE)
        table.add_column("Component", style="cyan")
        table.add_column("Status")
        table.add_row("Qdrant", "Available" if status["qdrant_available"] else "Unavailable")
        table.add_row("SQLite registry", "Available" if status["sqlite_available"] else "Unavailable")
        table.add_row("Indexed documents", str(status["indexed_documents"]))
        table.add_row("Total source records", str(status["sources"]))
        table.add_row("Indexed chunks", str(status["chunks"]))
        table.add_row("Vectors", str(status["vectors"]))
        table.add_row("Collection", status["collection_name"])
        table.add_row("Embedding model", status["embedding_model"])
        table.add_row(
            "Embedding model availability",
            "Installed" if status["embedding_model_available"] else "Not installed / unavailable",
        )
        table.add_row("Vector dimensions", _value(status["vector_dimensions"]))
        table.add_row("Automatic retrieval", "On" if status["automatic_retrieval"] else "Off")
        table.add_row("Qdrant storage", status["qdrant_path"])
        table.add_row("SQLite registry", status["sqlite_path"])
        self.console.print(table)
        if status["embedding_model_missing"]:
            self.console.print(
                f"[yellow]RAG requires the configured embedding model. Install it "
                f"manually with: ollama pull {status['embedding_model']}[/yellow]"
            )
        if status["embedding_error"]:
            self.console.print(f"[dim]{status['embedding_error']}[/dim]")

    def show_sources(
        self, sources: list[SourceRecord], total: int, page: int, page_size: int
    ) -> None:
        table = Table(title="Indexed document sources", box=box.SIMPLE)
        table.add_column("ID", style="cyan")
        table.add_column("Filename")
        table.add_column("Type")
        table.add_column("Chunks", justify="right")
        table.add_column("Imported")
        table.add_column("Status")
        for source in sources:
            table.add_row(
                source.id[:8],
                source.filename,
                source.file_type or "Unknown",
                str(source.chunk_count),
                source.imported_at[:10],
                source.status,
            )
        if not sources:
            self.console.print("[dim]No source documents are registered.[/dim]")
        else:
            self.console.print(table)
            note = f"Page {page} · {total} source records · {page_size} per page."
            if page * page_size < total:
                note += f" Use /sources {page + 1} for the next page."
            self.console.print(f"[dim]{note}[/dim]")

    def show_source_details(self, source: SourceRecord) -> None:
        table = Table(title=f"Source · {source.filename}", box=box.SIMPLE)
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        for label, value in (
            ("ID", source.id),
            ("File type", source.file_type),
            ("Size", format_bytes(source.file_size)),
            ("Status", source.status),
            ("Chunks", str(source.chunk_count)),
            ("Imported", source.imported_at),
            ("Last indexed", _value(source.indexed_at)),
            ("Embedding model", _value(source.embedding_model)),
            ("Content SHA-256", source.content_hash),
            ("Error", _value(source.error)),
            ("Document metadata", json.dumps(source.metadata, ensure_ascii=False, indent=2)),
        ):
            table.add_row(label, Text(value))
        self.console.print(table)

    def show_search_results(self, chunks: tuple[RetrievedChunk, ...]) -> None:
        if not chunks:
            self.console.print("[dim]No matching document passages found.[/dim]")
            return
        self.console.print(f"[bold]Retrieved {len(chunks)} passage(s)[/bold]")
        for index, chunk in enumerate(chunks, start=1):
            payload = chunk.payload
            citation = Citation(
                label=index,
                filename=str(payload.get("filename") or "Unknown source"),
                page_number=payload.get("page_number")
                if isinstance(payload.get("page_number"), int)
                else None,
                section_heading=payload.get("section_heading")
                if isinstance(payload.get("section_heading"), str)
                else None,
                start_line=payload.get("start_line")
                if isinstance(payload.get("start_line"), int)
                else None,
                end_line=payload.get("end_line")
                if isinstance(payload.get("end_line"), int)
                else None,
            )
            table = Table.grid(padding=(0, 1))
            table.add_column()
            table.add_row(Text(citation.reference, style="cyan"))
            table.add_row(Text(f"Similarity: {chunk.score:.4f}", style="dim"))
            excerpt = str(payload.get("text", ""))
            table.add_row(Text(excerpt[:900] + ("…" if len(excerpt) > 900 else "")))
            self.console.print(Panel(table, border_style="blue"))

    def show_ingestion_progress(self, current: int, total: int, path: Any) -> None:
        self.console.print(
            f"[dim]Indexing {current}/{total}: {Path(path).name}[/dim]"
        )

    def show_ingestion_result(self, result: IngestionResult) -> None:
        prefix = f"{result.status.title()}"
        if result.source_id:
            prefix += f" · source {result.source_id[:8]}"
        style = "green" if result.status in {"indexed", "skipped"} else "yellow"
        self.console.print(f"[{style}]{prefix}: {result.message}[/{style}]")

    def show_folder_ingestion(
        self, files: list[Path], results: list[tuple[Path, IngestionResult]]
    ) -> None:
        if not files:
            self.console.print("[dim]No supported documents were found in that folder.[/dim]")
            return
        indexed = sum(result.status == "indexed" for _, result in results)
        skipped = sum(result.status == "skipped" for _, result in results)
        failed = sum(result.status == "failed" for _, result in results)
        self.console.print(
            f"Folder import complete: {len(files)} supported · "
            f"{indexed} indexed · {skipped} unchanged · {failed} failed."
        )
        for path, result in results:
            if result.status == "failed":
                self.console.print(
                    f"[yellow]{path.name}: {result.message}[/yellow]"
                )

    def show_remove_confirmation(self, source_id: str, filename: str) -> None:
        self.console.print(
            f"Remove indexed source [bold]{filename}[/bold] and its vectors? "
            f"Original file will remain untouched. Confirm with "
            f"[cyan]/remove-source {source_id} confirm[/cyan]."
        )

    def show_source_removed(self, filename: str) -> None:
        self.console.print(
            f"Removed indexed source [bold]{filename}[/bold]. Original file was not changed."
        )

    def show_remove_all_sources_confirmation(
        self, status: dict[str, Any]
    ) -> None:
        self.console.print(
            f"This will remove all {status['indexed_documents']} indexed source "
            f"record(s) and {status['chunks']} chunk(s) from LowP. Original files "
            f"will remain untouched. To proceed, enter "
            f"[cyan]/remove-all-sources confirm[/cyan]."
        )

    def show_all_sources_removed(self) -> None:
        self.console.print(
            "All indexed sources were removed from LowP; original files were not changed."
        )

    def show_reset_knowledge_confirmation(self, status: dict[str, Any]) -> None:
        self.console.print(
            f"This will delete LowP's {status['indexed_documents']} indexed "
            f"document record(s), {status['chunks']} chunk record(s), and the "
            f"configured Qdrant collection. Original files and chat history are "
            f"not affected. To proceed, enter [cyan]/reset-knowledge confirm[/cyan]."
        )

    def show_rag_mode(self, enabled: bool) -> None:
        self.console.print(
            f"Automatic document retrieval is {'on' if enabled else 'off'}."
        )

    def show_knowledge_reset(self) -> None:
        self.console.print("LowP indexed knowledge was reset; original files were not changed.")

    def show_error(self, message: str) -> None:
        self.console.print(f"[bold red]Error:[/bold red] {message}")

    def show_goodbye(self) -> None:
        self.console.print("[cyan]Goodbye.[/cyan]")

    def clear(self) -> None:
        self.console.clear()