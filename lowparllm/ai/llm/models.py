"""Small typed values exchanged by the local Ollama integration."""

from dataclasses import dataclass
from typing import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class GenerationMetrics:
    prompt_eval_count: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None

    @classmethod
    def from_response(cls, data: dict[str, object]) -> "GenerationMetrics":
        def metric(name: str) -> int | None:
            value = data.get(name)
            return value if isinstance(value, int) and not isinstance(value, bool) else None

        return cls(
            prompt_eval_count=metric("prompt_eval_count"),
            prompt_eval_duration_ns=metric("prompt_eval_duration"),
            eval_count=metric("eval_count"),
            eval_duration_ns=metric("eval_duration"),
            total_duration_ns=metric("total_duration"),
            load_duration_ns=metric("load_duration"),
        )

    @property
    def tokens_per_second(self) -> float | None:
        if self.eval_count is None or not self.eval_duration_ns:
            return None
        return self.eval_count / (self.eval_duration_ns / 1_000_000_000)


@dataclass(frozen=True)
class StreamEvent:
    content: str = ""
    done: bool = False
    metrics: GenerationMetrics | None = None


@dataclass(frozen=True)
class OllamaStatus:
    installed: bool
    connected: bool
    model_available: bool
    model_loaded: bool | None
    model_size: int | None = None
    model_details: str | None = None
    error: str | None = None
