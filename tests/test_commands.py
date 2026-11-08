import asyncio
from datetime import datetime
import unittest
from unittest.mock import Mock, patch

from lowparllm.cli.commands import CommandHandler, split_command
from lowparllm.core.config import AppConfig
from lowparllm.ai.rag.rag_service import IngestionResult, SearchResult
from lowparllm.ai.research.research_service import ResearchBundle
from lowparllm.ai.search.models import SearchResult as WebSearchResult, WebSource
from lowparllm.storage.database import SourceRecord
from lowparllm.ai.coding.workspace import CodingWorkspace
from pathlib import Path
import tempfile


STATUS = {
    "system": {"distribution": "Test Linux", "kernel": "test-kernel"},
    "cpu": {"model": "Test CPU"},
    "memory": {"total": 1024, "available": 512},
    "gpu": {"gpus": []},
    "software": {
        "python": "3.14",
        "ollama": {
            "installed": False,
            "service_available": False,
            "service_status": "Unavailable",
            "base_model_installed": False,
            "base_model": None,
            "model_details": "Unavailable",
        },
    },
}


class FakeRagService:
    automatic_retrieval = True

    def __init__(self) -> None:
        self.added_path = None
        self.knowledge_reset_count = 0

    async def status(self):
        return {
            "qdrant_available": True,
            "sqlite_available": True,
            "indexed_documents": 1,
            "chunks": 2,
        }

    async def add_file(self, path):
        self.added_path = path
        return IngestionResult("source-123", "indexed", "Indexed 2 chunks.")

    async def add_folder(self, _path, progress=None):
        del _path, progress
        return [], []

    async def list_sources(self, _page=1, page_size=20):
        del _page, page_size
        return [self.source()], 1

    async def get_source(self, source_id):
        return self.source() if self.source().id.startswith(source_id) else None

    async def search(self, _query):
        del _query
        return SearchResult(())

    async def remove_source(self, source_id):
        if not self.source().id.startswith(source_id):
            raise ValueError(f"Unknown source: {source_id}")
        return self.source()

    async def reindex(self, source_id):
        return IngestionResult(source_id, "indexed", "Reindexed.")

    async def reset_knowledge(self):
        self.knowledge_reset_count += 1

    async def set_automatic_retrieval(self, enabled):
        self.automatic_retrieval = enabled

    @staticmethod
    def source():
        return SourceRecord(
            id="source-123",
            filename="notes.md",
            original_path="/tmp/notes.md",
            file_type=".md",
            file_size=10,
            content_hash="hash",
            imported_at="2026-01-01",
            indexed_at="2026-01-01",
            status="indexed",
            chunk_count=2,
            error=None,
            embedding_model="fake",
            metadata={},
        )


class FakeResearchService:
    def __init__(self) -> None:
        self.settings = {
            "enabled": True,
            "automatic_research": False,
            "save_sources_enabled": True,
            "unlimited": False,
        }
        self.search_query = None
        self.opened_url = None
        self.saved_id = None

    def status(self):
        return {**self.settings, "provider": "fake"}

    @staticmethod
    def default_mode(question):
        if "my notes" in question and "current" in question:
            return "both"
        return "local" if "my notes" in question else "web"

    async def search(self, query, *, fresh=False, progress=None):
        del progress
        self.search_query = (query, fresh)
        return (
            WebSearchResult(
                title="Search result",
                url="https://example.com/",
                snippet="Snippet",
                domain="example.com",
            ),
        )

    async def open(self, url, *, progress=None):
        del progress
        self.opened_url = url
        return WebSource(
            source_id="1",
            title="Page",
            url=url,
            domain="example.com",
            published_date=None,
            accessed_at=datetime.now(),
            content="Readable page",
            evidence_type="extracted webpage",
        )

    async def save_source(self, source_id, *, progress=None):
        del progress
        self.saved_id = source_id
        return IngestionResult(source_id, "indexed", "Saved.")

    def set_setting(self, name, enabled):
        self.settings[name] = enabled

    def set_limit_setting(self, name, value):
        ranges = {
            "max_queries": (1, 5),
            "max_results": (1, 20),
            "max_pages": (1, 10),
            "timeout_seconds": (2, 60),
            "max_concurrent_requests": (1, 8),
            "max_page_bytes": (65536, 10485760),
            "max_page_chars": (1000, 50000),
            "max_context_chars": (2000, 60000),
            "search_cache_ttl_seconds": (0, 604800),
        }
        if name not in ranges:
            raise ValueError(f"Unknown web limit: {name}")
        minimum, maximum = ranges[name]
        parsed = float(value) if name == "timeout_seconds" else int(value)
        if not minimum <= parsed <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}.")
        self.settings[name] = parsed

    def set_search_option(self, name, value):
        self.settings[name] = value

    async def check_connection(self):
        return False, "SearXNG is not running."

    async def search_cache_stats(self):
        return {"entries": 0}

    async def clear_search_cache(self):
        return 2

    def should_research(self, _message):
        del _message
        return False

    async def research(self, question, mode="web", *, progress=None):
        del question, progress
        return ResearchBundle(mode, "evidence", (), ())

    async def research_web(self, question, *, citation_offset=0, progress=None):
        del question, citation_offset, progress
        return ResearchBundle("web", "web evidence", (), ())

    async def close(self):
        return None


class CommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = Mock(return_value=STATUS)
        self.rag = FakeRagService()
        self.research = FakeResearchService()
        self.handler = CommandHandler(
            AppConfig(),
            self.provider,
            rag_service=self.rag,
            research_service=self.research,
        )

    def execute(self, command: str):
        return asyncio.run(self.handler.execute(command))

    def test_windows_command_parsing_preserves_paths_and_quotes(self) -> None:
        self.assertEqual(
            split_command(r"/cd C:\Users\User\Projects", windows=True),
            ["/cd", r"C:\Users\User\Projects"],
        )
        self.assertEqual(
            split_command(r'/cd "C:\Users\User\My Projects\App"', windows=True),
            ["/cd", r"C:\Users\User\My Projects\App"],
        )
        self.assertEqual(
            split_command(r"/cd C:/Users/User/Projects", windows=True),
            ["/cd", "C:/Users/User/Projects"],
        )

    def test_all_commands_are_registered_and_dispatch(self) -> None:
        expected = {
            "/help": "help",
            "/status": "status",
            "/model": "model",
            "/router": "router_status",
            "/settings": "settings",
            "/clear": "clear",
            "/reset": "reset",
            "/exit": "exit",
        }
        for command, action in expected.items():
            with self.subTest(command=command):
                result = self.execute(command)
                self.assertEqual(result.action, action)

    def test_help_lists_all_commands(self) -> None:
        result = self.execute("/help")
        self.assertEqual(
            result.payload,
            (
                "/help",
                "/status",
                "/model",
                "/train",
                "/adapters",
                "/router",
                "/generation",
                "/code",
                "/pwd",
                "/ls",
                "/cd",
                "/tree",
                "/find",
                "/project",
                "/file",
                "/git",
                "/run",
                "/test",
                "/build",
                "/lint",
                "/review",
                "/debug",
                "/permissions",
                "/init",
                "/pdf",
                "/doctor",
                "/settings",
                "/clear",
                "/reset",
                "/exit",
                "/knowledge",
                "/add",
                "/add-folder",
                "/sources",
                "/source",
                "/search",
                "/remove-source",
                "/remove-all-sources",
                "/reindex",
                "/rag",
                "/reset-knowledge",
                "/web",
                "/web-search",
                "/web-open",
                "/research",
                "/research-local",
                "/research-web",
                "/research-both",
                "/web-save",
                "/web-settings",
                "/web-cache",
                "/web-cache-clear",
                "/web-cache-settings",
                "/copy",
            ),
        )

    def test_coding_mode_can_toggle_without_disabling_global_commands(self) -> None:
        enabled = self.execute("/code on")
        self.assertEqual(enabled.action, "code_status")
        self.assertTrue(self.handler.workspace.coding_mode)

        model = self.execute("/model")
        self.assertEqual(model.action, "model")

        disabled = self.execute("/code off")
        self.assertEqual(disabled.action, "code_status")
        self.assertFalse(self.handler.workspace.coding_mode)

    def test_search_subcommands_target_project_while_plain_search_keeps_rag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "app.py").write_text("def useful_symbol():\n    return 1\n", encoding="utf-8")
            self.handler.workspace = CodingWorkspace(root)

            project = self.execute("/search symbol useful_symbol")
            self.assertEqual(project.action, "project_search")
            self.assertIn("app.py:1", project.payload["matches"][0])

            knowledge = self.execute("/search useful_symbol")
            self.assertEqual(knowledge.action, "search")

    def test_status_and_model_use_live_read_only_status(self) -> None:
        status = self.execute("/status")
        model = self.execute("/model")
        self.assertIs(status.payload["status"], STATUS)
        self.assertEqual(status.payload["message_count"], 0)
        self.assertEqual(status.payload["ollama_url"], "http://127.0.0.1:11434")
        self.assertEqual(model.payload["config"].base_model, "qwen3.5:4b")
        self.assertIs(model.payload["status"], STATUS)
        self.assertEqual(self.provider.call_count, 2)

    def test_router_commands_show_plan_and_keep_unsupported_routes_disabled(self) -> None:
        plan = self.execute("/router plan compare my notes with the latest React docs")
        self.assertEqual(plan.action, "router_plan")
        self.assertEqual(plan.payload["plan"].research_mode, "both")
        self.assertEqual(plan.payload["model"], "qwen3.5:4b")
        self.assertEqual(self.execute("/router explain").action, "router_plan")
        with patch("lowparllm.cli.commands.persist_router_setting"):
            self.assertFalse(
                self.execute("/router auto off").payload["config"].automatic
            )
            self.assertTrue(
                self.execute("/router auto on").payload["config"].automatic
            )
        self.assertEqual(self.execute("/router model auto on").action, "error")
        self.assertEqual(self.execute("/router adapters auto on").action, "error")
        with patch("lowparllm.cli.commands.persist_router_setting"):
            self.assertEqual(
                self.execute("/router adapters auto off").action, "router_status"
            )
        self.assertEqual(self.execute("/router reset").action, "error")

    def test_router_capability_inventory_reports_real_backend_limits(self) -> None:
        result = self.execute("/router capabilities")
        self.assertEqual(result.action, "router_capabilities")
        inventory = {item.identifier: item for item in result.payload}
        self.assertEqual(inventory["adapter.inference"].status.value, "Incompatible")
        self.assertEqual(inventory["coding.workspace"].status.value, "Available")

    def test_settings_resolve_centralized_paths(self) -> None:
        result = self.execute("/settings")
        self.assertEqual(result.payload["config"].base_model, "qwen3.5:4b")
        self.assertTrue(str(result.payload["config_dir"]).endswith("/lowp"))
        self.assertTrue(str(result.payload["data_dir"]).endswith("/lowp"))
        self.assertTrue(str(result.payload["cache_dir"]).endswith("/lowp"))

    def test_doctor_reports_required_and_optional_components(self) -> None:
        self.handler.research_service = None
        result = self.execute("/doctor")
        self.assertEqual(result.action, "doctor")
        components = {check["component"] for check in result.payload["checks"]}
        self.assertIn("Python", components)
        self.assertIn("Required dependencies", components)
        self.assertIn("Ollama service", components)
        self.assertIn("SearXNG", components)
        self.assertIn("Local Qdrant store", components)
        self.assertIn("Launcher and PATH", components)

    def test_pdf_command_remains_read_only_and_rejects_creation(self) -> None:
        help_result = self.execute("/pdf help")
        self.assertEqual(help_result.action, "pdf_help")
        self.assertIn("inspect", help_result.payload)
        self.assertIn("extract-text", help_result.payload)
        self.assertIn("summarize", help_result.payload)

        rejected = self.execute('/pdf create "A report"')
        self.assertEqual(rejected.action, "error")
        self.assertNotIn("create", rejected.payload)

    def test_generation_command_updates_validated_runtime_preferences(self) -> None:
        with patch("lowparllm.cli.commands.persist_generation_setting") as persist:
            self.assertEqual(self.execute("/generation").action, "generation_settings")
            self.assertEqual(
                self.execute("/generation temperature 1.25").action,
                "generation_settings",
            )
            self.assertEqual(self.handler.config.temperature, 1.25)
            self.assertEqual(
                self.execute("/generation top-p 0.9").action,
                "generation_settings",
            )
            self.assertEqual(self.handler.config.top_p, 0.9)
            self.assertEqual(self.execute("/generation top_k 40").action, "generation_settings")
            self.assertEqual(self.handler.config.top_k, 40)
            self.assertEqual(
                self.execute('/generation keep-alive "10m"').action,
                "generation_settings",
            )
            self.assertEqual(self.handler.config.keep_alive, "10m")

        self.assertEqual(
            [call.args for call in persist.call_args_list],
            [
                ("temperature", 1.25),
                ("top_p", 0.9),
                ("top_k", 40),
                ("keep_alive", "10m"),
            ],
        )

    def test_generation_command_rejects_invalid_values_without_changing_config(self) -> None:
        with patch("lowparllm.cli.commands.persist_generation_setting"):
            for command in (
                "/generation temperature 2.1",
                "/generation temperature nan",
                "/generation top_p -0.1",
                "/generation top_k 1.5",
                "/generation top_k 0",
                '/generation keep_alive " "',
                "/generation unknown 1",
                "/generation temperature",
            ):
                with self.subTest(command=command):
                    self.assertEqual(self.execute(command).action, "error")
        self.assertEqual(self.handler.config.temperature, 0.7)
        self.assertEqual(self.handler.config.top_p, 0.8)
        self.assertEqual(self.handler.config.top_k, 20)
        self.assertEqual(self.handler.config.keep_alive, "5m")

    def test_regular_text_is_not_a_command(self) -> None:
        self.assertIsNone(self.execute("hello"))
        self.assertIsNone(self.execute("  hello"))
        self.assertEqual(self.provider.call_count, 0)

    def test_invalid_commands_and_arguments_are_helpful(self) -> None:
        unknown = self.execute("/wat")
        self.assertEqual(unknown.action, "error")
        self.assertIn("Unknown command", unknown.payload)
        arguments = self.execute("/help now")
        self.assertEqual(arguments.action, "error")
        self.assertIn("does not accept arguments", arguments.payload)

    def test_malformed_command_is_reported(self) -> None:
        result = self.execute('/help "')
        self.assertEqual(result.action, "error")
        self.assertIn("Could not parse command", result.payload)

    def test_knowledge_commands_dispatch_and_paths_with_spaces(self) -> None:
        commands = {
            "/knowledge": "knowledge",
            '/add "/tmp/notes with spaces.md"': "ingestion",
            '/add-folder "/tmp/my notes"': "folder_ingestion",
            "/sources 2": "sources",
            "/source source-123": "source",
            "/search explain the network": "search",
            "/remove-source source-123": "remove_source_confirm",
            "/remove-all-sources": "remove_all_sources_confirm",
            "/reindex source-123": "ingestion",
            "/rag status": "rag",
            "/reset-knowledge": "reset_knowledge_confirm",
        }
        for text, action in commands.items():
            with self.subTest(command=text):
                self.assertEqual(self.execute(text).action, action)
        self.assertEqual(str(self.rag.added_path), "/tmp/notes with spaces.md")

    def test_rag_toggle_and_destructive_confirmation(self) -> None:
        self.assertEqual(self.execute("/rag off").payload["enabled"], False)
        self.assertEqual(self.execute("/rag on").payload["enabled"], True)
        self.assertEqual(
            self.execute("/reset-knowledge confirm").action, "knowledge_reset"
        )
        self.assertEqual(
            self.execute("/remove-source source-123 confirm").action, "source_removed"
        )
        self.assertEqual(
            self.execute("/remove-all-sources confirm").action,
            "all_sources_removed",
        )
        self.assertEqual(self.rag.knowledge_reset_count, 2)
        self.assertEqual(
            self.execute("/remove-all-sources invalid").action, "error"
        )
        self.assertEqual(
            self.execute('/remove-source "unmatched id" confirm').action, "error"
        )

    def test_web_and_research_commands_dispatch(self) -> None:
        self.assertEqual(self.execute("/web").action, "web_status")
        searched = self.execute("/web-search latest React release")
        self.assertEqual(searched.action, "web_search")
        self.assertEqual(self.research.search_query, ("latest React release", False))
        fresh = self.execute("/web-search --fresh latest React release")
        self.assertEqual(fresh.action, "web_search")
        self.assertEqual(self.research.search_query, ("latest React release", True))
        opened = self.execute("/web-open https://example.com/")
        self.assertEqual(opened.action, "web_page")
        self.assertEqual(self.research.opened_url, "https://example.com/")
        self.assertEqual(
            self.execute("/research compare my notes with current React").payload,
            {"question": "compare my notes with current React", "mode": "both"},
        )
        self.assertEqual(
            self.execute("/research-local local question").payload["mode"], "local"
        )
        self.assertEqual(
            self.execute("/research-web current question").payload["mode"], "web"
        )
        self.assertEqual(
            self.execute("/research-both combined question").payload["mode"], "both"
        )
        self.assertEqual(
            self.execute("/research local whats frontend development?").payload,
            {"question": "whats frontend development?", "mode": "local"},
        )
        self.assertEqual(
            self.execute("/research WEB latest React release").payload,
            {"question": "latest React release", "mode": "web"},
        )
        self.assertEqual(
            self.execute("/research both compare my notes with current docs").payload,
            {"question": "compare my notes with current docs", "mode": "both"},
        )
        self.assertEqual(
            self.execute("/research local").action,
            "error",
        )
        self.assertEqual(self.execute("/web-save 1").action, "web_saved")
        self.assertEqual(self.research.saved_id, "1")

    def test_web_settings_commands_validate_and_update_runtime_settings(self) -> None:
        self.assertEqual(self.execute("/web-settings").action, "web_settings")
        updated = self.execute("/web-settings auto on")
        self.assertEqual(updated.action, "web_settings")
        self.assertTrue(self.research.settings["automatic_research"])
        updated = self.execute("/web-settings unlimited on")
        self.assertEqual(updated.action, "web_settings")
        self.assertTrue(self.research.settings["unlimited"])
        updated = self.execute("/web-settings max_pages 3")
        self.assertEqual(updated.action, "web_settings")
        self.assertEqual(updated.payload["max_pages"], 3)
        self.assertEqual(self.research.settings["max_pages"], 3)
        self.assertEqual(self.execute("/web-settings max_pages 11").action, "error")
        self.assertEqual(
            self.execute("/web-settings timeout_seconds 12.5").action,
            "web_settings",
        )
        self.assertEqual(self.research.settings["timeout_seconds"], 12.5)
        self.assertEqual(self.execute("/web-settings invalid on").action, "error")
        self.assertEqual(self.execute("/web-open").action, "error")
        self.assertEqual(self.execute("/web-search").action, "error")
        self.assertEqual(self.execute("/copy").action, "copy_code_usage")
        self.assertEqual(self.execute("/copy 2").payload, 2)
        self.assertEqual(self.execute("/copy 0").action, "error")
        self.assertEqual(self.execute("/copy one").action, "error")
        self.assertEqual(self.execute("/web-cache").action, "web_cache")
        self.assertEqual(self.execute("/web-cache-clear").action, "web_cache_cleared")
        self.assertEqual(self.execute("/web-cache-settings ttl 120").action, "web_cache_settings")


if __name__ == "__main__":
    unittest.main()
