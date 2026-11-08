import asyncio
from datetime import datetime, timezone
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from lowparllm.ai.llm.models import GenerationMetrics, StreamEvent
from lowparllm.ai.llm.model_manager import OllamaModel
from lowparllm.ai.rag.citations import Citation
from lowparllm.ai.rag.rag_service import ChatContext
from lowparllm.ai.research.citations import WebCitation
from lowparllm.ai.research.research_service import ResearchBundle
from lowparllm.cli.renderer import Renderer
from lowparllm.cli.terminal import TerminalApp
from lowparllm.core.config import AppConfig, RagConfig
from tests.test_commands import STATUS


class FakeInput:
    def __init__(self, messages):
        self.messages = iter(messages)

    async def read_message(self):
        try:
            message = next(self.messages)
        except StopIteration:
            raise EOFError from None
        if isinstance(message, BaseException):
            raise message
        return message


class FakeRenderer:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, args))

        return record


class FakeClient:
    async def aclose(self):
        return None


class FakeChatService:
    def __init__(self, config, client):
        self.config = config
        self.client = client
        self.message_count = 0
        self.last_user_message = None
        self.retrieved_context = None
        self.ephemeral_calls = []
        self.conversation = type("ConversationState", (), {"last_pruned_messages": 0})()

    async def stream_response(
        self,
        user_message,
        retrieved_context=None,
        response_guidance=None,
        *,
        ephemeral=False,
    ):
        self.last_user_message = user_message
        self.retrieved_context = retrieved_context
        self.response_guidance = response_guidance
        self.ephemeral_calls.append(ephemeral)
        yield StreamEvent(content="real streamed chunk")
        yield StreamEvent(done=True, metrics=GenerationMetrics(eval_count=3))
        if not ephemeral:
            self.message_count += 2

    def reset(self):
        self.message_count = 0


class BlockingChatService(FakeChatService):
    def __init__(self, config, client):
        super().__init__(config, client)
        self.started = asyncio.Event()

    async def stream_response(self, user_message, *args, ephemeral=False, **kwargs):
        del args, kwargs, ephemeral
        self.last_user_message = user_message
        self.started.set()
        yield StreamEvent(content="partial")
        await asyncio.Event().wait()


class FakeRagService:
    def __init__(self, *, has_documents=False, automatic_retrieval=False):
        self.has_documents = has_documents
        self.automatic_retrieval = automatic_retrieval

    async def has_indexed_sources(self):
        return self.has_documents and self.automatic_retrieval

    async def context_for_chat(self, _query):
        del _query
        return ChatContext(
            "untrusted retrieved context",
            (Citation(1, "notes.md", None, "Heading", None, None),),
            (),
        )

    async def close(self):
        return None


class FakeResearchService:
    def __init__(self, *, automatic=False):
        self.automatic = automatic
        self.questions = []
        self.fresh_flags = []

    def should_research(self, _message):
        del _message
        return self.automatic

    async def research_web(self, question, *, citation_offset=0, progress=None):
        self.questions.append(question)
        if progress is not None:
            progress("Searching the web")
        return ResearchBundle(
            "web",
            "untrusted web evidence",
            (
                WebCitation(
                    label=citation_offset + 1,
                    title="Source",
                    domain="example.com",
                    url="https://example.com/",
                    published_date=None,
                    accessed_at=datetime.now(timezone.utc),
                    evidence_type="extracted webpage",
                    relevant_excerpt="Supported finding.",
                ),
            ),
            (),
        )

    async def research(self, question, mode="web", *, fresh=False, progress=None):
        self.questions.append(question)
        self.fresh_flags.append(fresh)
        if progress is not None:
            progress("Searching local knowledge")
        return ResearchBundle(mode, "untrusted research evidence", (), ())

    async def close(self):
        return None


class TerminalTests(unittest.TestCase):
    def test_startup_commands_and_exit(self) -> None:
        renderer = FakeRenderer()
        with patch("lowparllm.cli.terminal.ModelManager") as manager_factory:
            manager = manager_factory.return_value
            model = OllamaModel("qwen3.5:4b", None, None, None)
            manager.list_models = AsyncMock(return_value=(model,))
            manager.model_info = AsyncMock(return_value=model)
            manager.close = AsyncMock()
            app = TerminalApp(
                AppConfig(),
                renderer=renderer,
                input_handler=FakeInput(
                    [
                        "/help",
                        "/status",
                        "/model",
                        "/settings",
                        "/clear",
                        "/reset",
                        "/unknown",
                        "/exit",
                    ]
                ),
                status_provider=lambda: STATUS,
                ollama_client=FakeClient(),
                chat_service=FakeChatService(AppConfig(), FakeClient()),
                rag_service=FakeRagService(),
            )
            self.assertEqual(asyncio.run(app.run()), 0)
        names = [name for name, _ in renderer.calls]
        self.assertEqual(
            names,
            [
                "show_startup",
                "show_workspace",
                "show_help",
                "show_status",
                "show_model_overview",
                "show_settings",
                "clear",
                "show_reset",
                "show_error",
                "show_goodbye",
            ],
        )

    def test_normal_messages_render_actual_stream_events(self) -> None:
        renderer = FakeRenderer()
        app = TerminalApp(
            AppConfig(),
            renderer=renderer,
            input_handler=FakeInput(["What is the answer?", "/exit"]),
            status_provider=lambda: STATUS,
            ollama_client=FakeClient(),
            chat_service=FakeChatService(AppConfig(), FakeClient()),
            rag_service=FakeRagService(),
        )
        asyncio.run(app.run())
        self.assertIn(("begin_stream", ()), renderer.calls)
        self.assertIn(("stream_token", ("real streamed chunk",)), renderer.calls)
        self.assertTrue(any(name == "finish_stream" for name, _ in renderer.calls))

    def test_coding_mode_uses_agent_and_returns_to_normal_chat_without_restart(self) -> None:
        renderer = FakeRenderer()
        chat = FakeChatService(AppConfig(), FakeClient())
        app = TerminalApp(
            AppConfig(),
            renderer=renderer,
            input_handler=FakeInput(
                [
                    "/code on",
                    "Explain this project",
                    "/settings",
                    "/code off",
                    "Normal assistant again",
                    "/exit",
                ]
            ),
            status_provider=lambda: STATUS,
            ollama_client=FakeClient(),
            chat_service=chat,
            rag_service=FakeRagService(),
        )

        asyncio.run(app.run())

        self.assertEqual(chat.ephemeral_calls, [True, False])
        self.assertEqual(chat.message_count, 2)
        names = [name for name, _ in renderer.calls]
        self.assertIn("show_agent_outcome", names)
        self.assertIn("show_settings", names)
        self.assertIn("begin_stream", names)

    def test_ctrl_c_cancels_generation_and_returns_to_prompt(self) -> None:
        async def run() -> tuple[int, FakeRenderer]:
            config = AppConfig()
            client = FakeClient()
            chat = BlockingChatService(config, client)
            renderer = FakeRenderer()
            app = TerminalApp(
                config,
                renderer=renderer,
                input_handler=FakeInput(["question", "/exit"]),
                status_provider=lambda: STATUS,
                ollama_client=client,
                chat_service=chat,
                rag_service=FakeRagService(),
            )
            task = asyncio.create_task(app.run())
            await chat.started.wait()
            task.cancel()
            return await task, renderer

        result, renderer = asyncio.run(run())
        self.assertEqual(result, 0)
        names = [name for name, _ in renderer.calls]
        self.assertIn("show_generation_stopped", names)
        self.assertNotIn("finish_stream", names)
        self.assertIn("show_goodbye", names)

    def test_ctrl_c_cancels_active_coding_task(self) -> None:
        async def run() -> tuple[int, FakeRenderer, TerminalApp]:
            config = AppConfig()
            client = FakeClient()
            chat = BlockingChatService(config, client)
            renderer = FakeRenderer()
            app = TerminalApp(
                config,
                renderer=renderer,
                input_handler=FakeInput(["/code on", "coding task"]),
                status_provider=lambda: STATUS,
                ollama_client=client,
                chat_service=chat,
                rag_service=FakeRagService(),
            )
            task = asyncio.create_task(app.run())
            await chat.started.wait()
            task.cancel()
            return await task, renderer, app

        result, renderer, app = asyncio.run(run())
        self.assertEqual(result, 0)
        self.assertFalse(app.coding_agent.stop())
        self.assertIn(
            "show_generation_stopped",
            [name for name, _ in renderer.calls],
        )

    def test_ctrl_c_and_eof_exit_cleanly(self) -> None:
        for ending in (KeyboardInterrupt(), EOFError()):
            renderer = FakeRenderer()
            app = TerminalApp(
                AppConfig(),
                renderer=renderer,
                input_handler=FakeInput([ending]),
                status_provider=lambda: STATUS,
                ollama_client=FakeClient(),
                rag_service=FakeRagService(),
            )
            self.assertEqual(asyncio.run(app.run()), 0)
            self.assertIn("show_goodbye", [name for name, _ in renderer.calls])

    def test_retrieved_context_and_citations_flow_through_terminal(self) -> None:
        renderer = FakeRenderer()
        config = AppConfig(rag=RagConfig(automatic_retrieval=True))
        chat = FakeChatService(config, FakeClient())
        app = TerminalApp(
            config,
            renderer=renderer,
            input_handler=FakeInput(["question", "/exit"]),
            status_provider=lambda: STATUS,
            ollama_client=FakeClient(),
            chat_service=chat,
            rag_service=FakeRagService(
                has_documents=True, automatic_retrieval=True
            ),
        )
        asyncio.run(app.run())
        self.assertEqual(chat.retrieved_context, "untrusted retrieved context")
        self.assertIn("show_citations", [name for name, _ in renderer.calls])

    def test_casual_chat_does_not_retrieve_when_automatic_rag_is_off(self) -> None:
        renderer = FakeRenderer()
        chat = FakeChatService(AppConfig(), FakeClient())
        app = TerminalApp(
            AppConfig(),
            renderer=renderer,
            input_handler=FakeInput(["Hello", "/exit"]),
            status_provider=lambda: STATUS,
            ollama_client=FakeClient(),
            chat_service=chat,
            rag_service=FakeRagService(
                has_documents=True, automatic_retrieval=False
            ),
        )

        asyncio.run(app.run())

        self.assertIsNone(chat.retrieved_context)
        self.assertNotIn("show_rag_searching", [name for name, _ in renderer.calls])

    def test_explicit_research_uses_local_chat_and_citations(self) -> None:
        renderer = FakeRenderer()
        chat = FakeChatService(AppConfig(), FakeClient())
        research = FakeResearchService()
        app = TerminalApp(
            AppConfig(),
            renderer=renderer,
            input_handler=FakeInput(["/research-web current React changes", "/exit"]),
            status_provider=lambda: STATUS,
            ollama_client=FakeClient(),
            chat_service=chat,
            rag_service=FakeRagService(),
            research_service=research,
        )
        asyncio.run(app.run())
        self.assertEqual(research.questions, ["current React changes"])
        self.assertEqual(chat.retrieved_context, "untrusted research evidence")
        self.assertIn("show_citations", [name for name, _ in renderer.calls])

    def test_automatic_research_is_visible_and_uses_only_the_current_message(self) -> None:
        renderer = FakeRenderer()
        chat = FakeChatService(AppConfig(), FakeClient())
        research = FakeResearchService(automatic=True)
        app = TerminalApp(
            AppConfig(),
            renderer=renderer,
            input_handler=FakeInput(["What are the latest React changes?", "/exit"]),
            status_provider=lambda: STATUS,
            ollama_client=FakeClient(),
            chat_service=chat,
            rag_service=FakeRagService(),
            research_service=research,
        )
        asyncio.run(app.run())
        self.assertEqual(
            research.questions, ["What are the latest React changes?"]
        )
        self.assertIn("untrusted research evidence", chat.retrieved_context)
        stages = [
            args[1]
            for name, args in renderer.calls
            if name == "show_research_stage"
        ]
        self.assertIn("Generating final response", stages)

    def test_routed_research_without_evidence_does_not_guess(self) -> None:
        renderer = FakeRenderer()
        chat = FakeChatService(AppConfig(), FakeClient())
        research = FakeResearchService()
        research.research = AsyncMock(
            return_value=ResearchBundle("web", "", (), ())
        )
        app = TerminalApp(
            AppConfig(),
            renderer=renderer,
            input_handler=FakeInput(["What are the latest React changes?", "/exit"]),
            status_provider=lambda: STATUS,
            ollama_client=FakeClient(),
            chat_service=chat,
            rag_service=FakeRagService(),
            research_service=research,
        )

        asyncio.run(app.run())

        self.assertIsNone(chat.last_user_message)
        self.assertIn("show_research_no_evidence", [name for name, _ in renderer.calls])

    def test_status_renderer_handles_missing_hardware_and_ollama(self) -> None:
        from rich.console import Console

        console = Console(record=True, width=100, color_system=None)
        renderer = Renderer(console)
        renderer.show_status(
            {"status": STATUS, "message_count": 0, "ollama_url": "http://local", "model": "qwen3.5:4b"},
            "0.1.0",
        )
        rendered = console.export_text()
        self.assertIn("Unavailable", rendered)
        self.assertIn("LowParLLM 0.1.0", rendered)
        self.assertIn("Install/start Ollama", rendered)

    def test_help_renderer_keeps_all_commands_at_multiple_widths(self) -> None:
        from rich.console import Console

        commands = (
            "/help", "/status", "/model", "/copy", "/web",
            *(f"/future-command-{index}" for index in range(24)),
        )
        for width in (48, 96, 180):
            with self.subTest(width=width):
                console = Console(record=True, width=width, color_system=None)
                Renderer(console).show_help(commands)
                rendered = console.export_text()
                for command in commands:
                    self.assertIn(command, rendered)
                self.assertTrue(
                    all(len(line) <= width for line in rendered.splitlines()),
                    f"help output overflowed a {width}-column terminal",
                )

        from rich.console import Console

        wide = Console(record=True, width=180, color_system=None)
        Renderer(wide).show_help(("/help", "/status", "/copy", "/web"))
        general_rows = [
            line for line in wide.export_text().splitlines()
            if "/help" in line
        ]
        self.assertTrue(general_rows)
        self.assertIn("/copy", general_rows[0])

    def test_research_stage_label_reflects_selected_mode(self) -> None:
        from rich.console import Console

        console = Console(record=True, width=100, color_system=None)
        renderer = Renderer(console)
        for mode, label in (
            ("local", "Local research"),
            ("web", "Web research"),
            ("both", "Combined research"),
        ):
            renderer.show_research_stage(mode, "Searching sources")
            self.assertIn(f"{label}: Searching sources", console.export_text())

    def test_stream_metrics_use_ollama_durations(self) -> None:
        from rich.console import Console

        console = Console(record=True, width=100, color_system=None)
        Renderer(console).finish_stream(
            GenerationMetrics(
                prompt_eval_count=10,
                prompt_eval_duration_ns=1_500_000_000,
                eval_count=8,
                eval_duration_ns=2_000_000_000,
                total_duration_ns=4_000_000_000,
            )
        )
        rendered = console.export_text()
        self.assertIn("1.50s prompt eval", rendered)
        self.assertIn("2.00s generation", rendered)
        self.assertIn("4.00s total", rendered)

    def test_assistant_markdown_links_and_code_blocks_are_rendered_and_copyable(self):
        from rich.console import Console

        output = StringIO()
        console = Console(file=output, record=True, width=100)
        renderer = Renderer(console)
        renderer.begin_stream()
        renderer.stream_token(
            "Visit https://example.com/docs and [the guide]"
            "(https://example.com/guide).\n\n```bash\nlowp --help\n```"
        )
        renderer.finish_stream(None)

        self.assertIn("https://example.com/docs", output.getvalue())
        self.assertIn("lowp --help", output.getvalue())
        html = console.export_html()
        self.assertIn('href="https://example.com/docs"', html)
        self.assertIn('href="https://example.com/guide"', html)

        with (
            patch("lowparllm.cli.renderer.shutil.which", return_value="/usr/bin/wl-copy"),
            patch("lowparllm.cli.renderer.subprocess.run") as run,
        ):
            run.return_value.returncode = 0
            renderer.copy_code_block(1)
        self.assertEqual(run.call_args.kwargs["input"], b"lowp --help\n")
        with self.assertRaisesRegex(ValueError, "No code block 2"):
            renderer.copy_code_block(2)
        with (
            patch("lowparllm.cli.renderer.os", SimpleNamespace(name="nt")),
            patch("lowparllm.cli.renderer.shutil.which", return_value="clip.exe"),
            patch("lowparllm.cli.renderer.subprocess.run") as windows_run,
        ):
            windows_run.return_value.returncode = 0
            renderer.copy_code_block(1)
        self.assertEqual(windows_run.call_args.args[0], ["clip.exe"])
        self.assertEqual(windows_run.call_args.kwargs["input"], b"lowp --help\n")


if __name__ == "__main__":
    unittest.main()
