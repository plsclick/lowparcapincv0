"""Structured task-routing values shared by the router and terminal."""

from dataclasses import dataclass
from enum import StrEnum


class TaskIntent(StrEnum):
    GENERAL = "general_conversation"
    TECHNICAL = "technical_explanation"
    LOCAL_DOCUMENTS = "local_document_question"
    WEB_RESEARCH = "internet_research"
    COMBINED_RESEARCH = "combined_research"
    PROGRAMMING = "programming_question"
    CODING_TASK = "coding_task"
    DATA_ANALYSIS = "data_analysis"
    MODEL_MANAGEMENT = "model_management"
    ADAPTER_MANAGEMENT = "adapter_management"
    TRAINING = "training"
    DOCUMENT_SUMMARIZATION = "document_summarization"
    SETTINGS = "application_settings"
    MULTI_STEP = "multi_step"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TaskPlan:
    request: str
    intent: TaskIntent
    complexity: str
    confidence: float
    research_mode: str | None
    required_capabilities: tuple[str, ...]
    steps: tuple[str, ...]
    user_confirmation_required: bool = False
    notes: tuple[str, ...] = ()
