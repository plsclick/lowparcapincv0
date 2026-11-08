"""Bounded in-memory conversation context for one application run."""

from collections.abc import Sequence

from lowparllm.ai.llm.models import ChatMessage


class Conversation:
    """Keep only complete successful turns and prepare bounded request context."""

    def __init__(self, system_prompt: str, max_history_messages: int = 40) -> None:
        if max_history_messages < 2:
            raise ValueError("Conversation history must allow at least one turn.")
        self.system_prompt = system_prompt
        self.max_history_messages = max_history_messages
        self._history: list[ChatMessage] = []
        self.last_pruned_messages = 0

    @property
    def message_count(self) -> int:
        return len(self._history)

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        return tuple(self._history)

    def prepare_messages(
        self, user_message: str, num_ctx: int
    ) -> tuple[list[ChatMessage], int]:
        """Create a request without mutating state if inference fails or is cancelled."""
        budget_chars = max(512, (num_ctx - 128) * 4)
        messages = list(self._history)
        messages.append({"role": "user", "content": user_message})
        original_count = len(messages)
        while (
            len(messages) > self.max_history_messages
            or self._over_budget(messages, budget_chars, self.system_prompt)
        ):
            removable = len(messages) - 1
            if removable < 2:
                break
            del messages[:2]

        request = [
            {"role": "system", "content": self.system_prompt},
            *messages,
        ]
        return request, original_count - len(messages)

    def complete_turn(
        self,
        request_messages: Sequence[ChatMessage],
        assistant_message: str,
        num_ctx: int,
    ) -> None:
        if len(request_messages) < 2 or request_messages[-1]["role"] != "user":
            raise ValueError("A conversation turn must end with its user message.")
        messages = list(request_messages[1:])
        messages.append({"role": "assistant", "content": assistant_message})
        budget_chars = max(512, (num_ctx - 128) * 4)
        while (
            len(messages) > self.max_history_messages
            or self._over_budget(messages, budget_chars, self.system_prompt)
        ):
            removable = len(messages) - 2
            if removable < 2:
                break
            del messages[:2]
        self._history = messages
        self.last_pruned_messages = max(
            0, len(request_messages) - 1 + 1 - len(self._history)
        )

    def reset(self) -> None:
        self._history.clear()
        self.last_pruned_messages = 0

    @staticmethod
    def _over_budget(
        messages: Sequence[ChatMessage], budget_chars: int, system_prompt: str
    ) -> bool:
        return len(system_prompt) + sum(
            len(message["content"]) for message in messages
        ) > budget_chars
