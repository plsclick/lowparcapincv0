import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from lowparllm.cli.input_handler import InputHandler


class InputHandlerTests(unittest.TestCase):
    def test_multiline_input_uses_prompt_toolkit_async_prompt(self) -> None:
        session = type("Session", (), {})()
        session.prompt_async = AsyncMock(return_value="line one\nline two")
        with patch("lowparllm.cli.input_handler.PromptSession", return_value=session) as factory:
            handler = InputHandler()
            value = asyncio.run(handler.read_message())
        kwargs = factory.call_args.kwargs
        self.assertTrue(kwargs["multiline"])
        self.assertIn("key_bindings", kwargs)
        session.prompt_async.assert_awaited_once_with("You > ")
        self.assertEqual(value, "line one\nline two")


if __name__ == "__main__":
    unittest.main()
