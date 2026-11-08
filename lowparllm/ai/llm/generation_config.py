"""Validated Ollama generation options."""

from dataclasses import dataclass
import math

from lowparllm.core.config import AppConfig


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    top_p: float
    num_ctx: int
    num_predict: int
    keep_alive: str
    top_k: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if not math.isfinite(self.top_p) or not 0 <= self.top_p <= 1:
            raise ValueError("Top-p must be between 0 and 1.")
        if not 256 <= self.num_ctx <= 131072:
            raise ValueError("Context size must be between 256 and 131072 tokens.")
        if self.num_predict != -1 and self.num_predict < 1:
            raise ValueError("Output token limit must be -1 or a positive integer.")
        if self.top_k < 1:
            raise ValueError("Top-k must be a positive integer.")
        if not self.keep_alive.strip():
            raise ValueError("Model keep-alive duration must not be empty.")

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "GenerationConfig":
        return cls(
            temperature=config.temperature,
            top_p=config.top_p,
            num_ctx=config.context_size,
            num_predict=config.num_predict,
            keep_alive=config.keep_alive,
            top_k=config.top_k,
        )

    def as_options(self) -> dict[str, float | int]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
