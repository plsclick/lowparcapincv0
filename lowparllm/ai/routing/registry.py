"""Inventory of implemented capabilities and their current operational status."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from lowparllm.core.config import AppConfig


class CapabilityStatus(StrEnum):
    AVAILABLE = "Available"
    UNAVAILABLE = "Unavailable"
    DISABLED = "Disabled"
    CONFIGURATION = "Requires configuration"
    INSTALLATION = "Requires installation"
    INCOMPATIBLE = "Incompatible"
    FAILED = "Failed health check"


@dataclass(frozen=True)
class Capability:
    identifier: str
    name: str
    description: str
    status: CapabilityStatus
    side_effects: bool = False
    confirmation_required: bool = False
    detail: str = ""


def capability_inventory(
    config: AppConfig,
    *,
    ollama_status: dict[str, Any] | None = None,
    searxng_status: tuple[bool, str | None] | None = None,
    rag_health: dict[str, Any] | None = None,
    adapters: tuple[dict[str, Any], ...] = (),
    missing_training_dependencies: tuple[str, ...] = (),
) -> tuple[Capability, ...]:
    ollama = (ollama_status or {}).get("software", {}).get("ollama", {})
    ollama_available = bool(ollama.get("service_available"))
    inference_status = (
        CapabilityStatus.AVAILABLE if ollama_available else CapabilityStatus.UNAVAILABLE
    )
    web_status = (
        CapabilityStatus.DISABLED
        if not config.web.enabled
        else CapabilityStatus.CONFIGURATION
        if searxng_status is None
        else CapabilityStatus.AVAILABLE
        if searxng_status[0]
        else CapabilityStatus.FAILED
    )
    if not config.rag.enabled:
        local_rag_status = CapabilityStatus.DISABLED
    elif rag_health is not None and (
        not rag_health.get("qdrant_available", False)
        or not rag_health.get("sqlite_available", False)
    ):
        local_rag_status = CapabilityStatus.FAILED
    elif rag_health is not None and not rag_health.get(
        "embedding_model_available", False
    ):
        local_rag_status = (
            CapabilityStatus.INSTALLATION
            if rag_health.get("embedding_model_missing", False)
            else CapabilityStatus.FAILED
        )
    else:
        local_rag_status = CapabilityStatus.AVAILABLE
    research_status = (
        CapabilityStatus.AVAILABLE
        if local_rag_status == CapabilityStatus.AVAILABLE
        or web_status == CapabilityStatus.AVAILABLE
        else CapabilityStatus.UNAVAILABLE
    )
    training_status = (
        CapabilityStatus.INSTALLATION
        if missing_training_dependencies
        else CapabilityStatus.CONFIGURATION
    )
    return (
        Capability("ollama.inference", "Ollama inference", "Local conversational model.", inference_status, detail=config.base_model),
        Capability("model.management", "Ollama model management", "Discover, inspect, select, and explicitly pull Ollama models.", inference_status, side_effects=True, confirmation_required=True),
        Capability(
            "adapter.management",
            "PEFT adapter management",
            "Inspect and manage LowP-stored adapters; inference activation is unsupported.",
            CapabilityStatus.AVAILABLE,
            side_effects=True,
            confirmation_required=True,
            detail=f"{len(adapters)} stored adapter(s); Ollama chat cannot load the current PEFT artifacts.",
        ),
        Capability(
            "adapter.inference",
            "PEFT adapter inference",
            "Dynamically loading these PEFT artifacts into Ollama chat.",
            CapabilityStatus.INCOMPATIBLE,
            detail="No PEFT-capable chat inference backend is implemented.",
        ),
        Capability(
            "rag.retrieval",
            "Local RAG",
            "Embedded Qdrant retrieval and SQLite document registry.",
            local_rag_status,
            side_effects=True,
            detail=(
                str(rag_health.get("embedding_error") or "Automatic retrieval follows [rag].automatic_retrieval.")
                if rag_health is not None
                else "Automatic retrieval follows [rag].automatic_retrieval."
            ),
        ),
        Capability("search.searxng", "SearXNG search", "Self-hosted JSON search provider.", web_status, detail=config.web.searxng_base_url),
        Capability("web.extraction", "Webpage extraction", "Safe webpage extraction for public URLs.", web_status),
        Capability("research.orchestration", "Research orchestration", "Local, web, and combined cited research.", research_status),
        Capability("search.cache", "Search cache", "Local SQLite search-result cache.", web_status),
        Capability("training.jobs", "LoRA/QLoRA training", "Optional local Transformers/PEFT background training.", training_status, side_effects=True, confirmation_required=True, detail=", ".join(missing_training_dependencies)),
        Capability("training.datasets", "Training datasets", "Explicit validation and import of user-selected datasets.", CapabilityStatus.AVAILABLE, side_effects=True),
        Capability(
            "coding.workspace",
            "Project-aware coding workspace",
            "Session-scoped project inspection and guarded file operations; command execution requires exact approval.",
            CapabilityStatus.AVAILABLE,
            side_effects=True,
            confirmation_required=True,
            detail="Filesystem boundaries are application-level restrictions, not an OS sandbox.",
        ),
        Capability("data.analysis", "Data analysis tools", "No dedicated data-analysis execution tool is implemented.", CapabilityStatus.UNAVAILABLE),
        Capability("application.settings", "Settings management", "Existing slash commands manage supported settings.", CapabilityStatus.AVAILABLE, side_effects=True, confirmation_required=True),
        Capability(
            "document.processing",
            "PDF reading",
            "Extract, inspect, and summarize text from local PDFs.",
            CapabilityStatus.AVAILABLE,
            detail="Uses the Phase 3 pypdf document-ingestion pipeline.",
        ),
    )
