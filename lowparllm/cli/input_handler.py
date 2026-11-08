"""Asynchronous, multiline terminal input."""

from collections.abc import Callable

from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit import PromptSession


class InputHandler:
    """Read multiline messages without blocking the asyncio event loop."""

    def __init__(self, prompt_provider: Callable[[], str] | None = None) -> None:
        self.prompt_provider = prompt_provider or (lambda: "You > ")
        bindings = KeyBindings()

        @bindings.add(
            "enter",
            filter=Condition(
                lambda: get_app().current_buffer.text.lstrip().startswith("/")
            ),
        )
        def submit_command(event) -> None:
            event.current_buffer.validate_and_handle()

        self._session: PromptSession[str] = PromptSession(
            multiline=True,
            key_bindings=bindings,
        )

    async def read_message(self) -> str:
        """Read multiline messages; Enter submits slash commands and Esc+Enter messages."""
        return await self._session.prompt_async(self.prompt_provider())