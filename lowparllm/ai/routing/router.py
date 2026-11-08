"""Conservative deterministic task routing over implemented LowP capabilities."""

import re

from lowparllm.ai.routing.models import TaskIntent, TaskPlan

_LOCAL = re.compile(
    r"\b(my|our|local|imported|indexed)\s+(?:[\w.-]+\s+){0,3}"
    r"(notes?|documents?|files?|pdfs?|knowledge|database|data)\b|"
    r"\b(the|that|this)\s+(pdf|document|file|notes?)\s+i\s+(added|imported)\b|"
    r"\b(pdf|document|file|notes?)\s+i\s+(added|imported)\b|"
    r"\bin\s+my\s+(project|notes|documents|files)\b",
    re.IGNORECASE,
)
_WEB = re.compile(
    r"\b(search(?:\s+the)?\s+web|look\s+(?:it\s+)?up\s+online|"
    r"on\s+the\s+internet|online sources?|current(?:ly)?|latest|"
    r"recent(?:ly)?|today|this week|this month|news|as of)\b",
    re.IGNORECASE,
)
_PROGRAMMING = re.compile(
    r"\b(what is|explain|how does|why does|difference between)\b.*"
    r"\b(python|javascript|typescript|react|css|html|sql|api|compiler|"
    r"programming|algorithm|exception|function|class)\b",
    re.IGNORECASE,
)
_CODING_TASK = re.compile(
    r"\b(write|create|implement|fix|debug|refactor|build|change)\b.*"
    r"\b(code|function|script|program|app|website|component|class|"
    r"repository|project|file|bug|feature)\b",
    re.IGNORECASE,
)
_MULTI_STEP = re.compile(
    r"\b(then|after that|and then|step[- ]by[- ]step|multi[- ]step)\b|"
    r"\b(research|compare|analy[sz]e|prepare|create)\b.*\b(report|plan|summary)\b",
    re.IGNORECASE,
)
_MANAGEMENT_PATTERNS = (
    (TaskIntent.MODEL_MANAGEMENT, re.compile(r"\b(list|switch|select|use|pull|download|inspect|show)\b.*\b(model|ollama)\b|\b(model|ollama)\b.*\b(list|switch|select|use|pull|download|inspect|show)\b", re.I), "model.management"),
    (TaskIntent.ADAPTER_MANAGEMENT, re.compile(r"\b(list|inspect|show|load|unload|activate|delete)\b.*\b(adapter|peft)\b|\b(adapter|peft)\b.*\b(list|inspect|show|load|unload|activate|delete)\b", re.I), "adapter.management"),
    (TaskIntent.TRAINING, re.compile(r"\b(train|training|fine[- ]?tun(?:e|ing)|dataset)\b", re.I), "training.jobs"),
    (TaskIntent.SETTINGS, re.compile(r"\b(settings|configuration|preferences)\b", re.I), "application.settings"),
    (TaskIntent.DATA_ANALYSIS, re.compile(r"\b(analy[sz]e|analysis|chart|statistics|dataset)\b", re.I), "data.analysis"),
)


class TaskRouter:
    """Create bounded plans without allowing retrieved content to affect policy."""

    def __init__(self, *, max_steps: int = 8) -> None:
        self.max_steps = max_steps

    def plan(self, request: str, *, forced_mode: str | None = None) -> TaskPlan:
        text = " ".join(request.split()).strip()
        if not text:
            return TaskPlan(
                request=text,
                intent=TaskIntent.UNKNOWN,
                complexity="unknown",
                confidence=1.0,
                research_mode=None,
                required_capabilities=("ollama.inference",),
                steps=("Ask the user to provide a request.",),
                notes=("The request is empty.",),
            )

        has_local = bool(_LOCAL.search(text))
        has_web = bool(_WEB.search(text))
        if forced_mode is not None:
            if forced_mode not in {"local", "web", "both"}:
                raise ValueError("Research mode must be local, web, or both.")
            has_local = forced_mode in {"local", "both"}
            has_web = forced_mode in {"web", "both"}

        document_summary = bool(
            re.search(r"\b(summarize|summary of|extract text from|inspect)\b", text, re.I)
            and re.search(r"\b(pdf|document)\b", text, re.I)
        )

        if document_summary:
            intent, mode = TaskIntent.DOCUMENT_SUMMARIZATION, "local" if has_local else None
            capabilities = ("document.processing", "ollama.inference")
            steps = ("Read the selected local PDF through the existing document extraction pipeline.", "Summarize only extracted content.")
        elif has_local and has_web:
            intent, mode = TaskIntent.COMBINED_RESEARCH, "both"
            capabilities = ("rag.retrieval", "search.searxng", "research.orchestration")
            steps = ("Retrieve relevant local passages.", "Search and extract web sources.", "Generate a cited comparison.")
        elif has_local:
            intent, mode = TaskIntent.LOCAL_DOCUMENTS, "local"
            capabilities = ("rag.retrieval", "research.orchestration")
            steps = ("Retrieve relevant local passages.", "Generate a cited answer from local sources.")
        elif any(pattern.search(text) for _, pattern, _ in _MANAGEMENT_PATTERNS):
            intent, capability = next(
                (intent, capability)
                for intent, pattern, capability in _MANAGEMENT_PATTERNS
                if pattern.search(text)
            )
            mode = None
            capabilities = (capability,)
            steps = (f"Use the existing {intent.value.replace('_', ' ')} commands; do not make changes without explicit confirmation.",)
        elif has_web:
            intent, mode = TaskIntent.WEB_RESEARCH, "web"
            capabilities = ("search.searxng", "web.extraction", "research.orchestration")
            steps = ("Search SearXNG.", "Extract relevant pages.", "Generate a cited answer.")
        elif _MULTI_STEP.search(text):
            intent, mode = TaskIntent.MULTI_STEP, None
            capabilities = ("ollama.inference",)
            steps = ("Handle the request as a bounded conversational plan.",)
        elif _CODING_TASK.search(text):
            intent, mode = TaskIntent.CODING_TASK, None
            capabilities = ("ollama.inference",)
            steps = ("Explain or draft code with the configured model; no project-editing tool is available.",)
        elif _PROGRAMMING.search(text):
            intent, mode = TaskIntent.PROGRAMMING, None
            capabilities = ("ollama.inference",)
            steps = ("Answer using the configured local model.",)
        elif re.search(r"\b(technical|technology|explain|how does|why does)\b", text, re.I):
            intent, mode = TaskIntent.TECHNICAL, None
            capabilities = ("ollama.inference",)
            steps = ("Answer using the configured local model.",)
        else:
            intent, mode = TaskIntent.GENERAL, None
            capabilities = ("ollama.inference",)
            steps = ("Answer directly with the configured local model.",)

        is_multi_step = bool(_MULTI_STEP.search(text))
        notes: list[str] = []
        if intent in {TaskIntent.CODING_TASK, TaskIntent.MULTI_STEP}:
            notes.append(
                "LowP does not execute shell commands or modify project files in this phase."
            )
        return TaskPlan(
            request=text,
            intent=intent,
            complexity="multi-step" if is_multi_step else "standard",
            confidence=0.95 if mode else 0.9,
            research_mode=mode,
            required_capabilities=capabilities,
            steps=tuple(steps[: self.max_steps]),
            notes=tuple(notes),
        )
