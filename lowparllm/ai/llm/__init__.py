"""Ollama-backed local conversational model integration."""

from lowparllm.ai.llm.chat_service import ChatService
from lowparllm.ai.llm.ollama_client import OllamaClient

__all__ = ["ChatService", "OllamaClient"]
