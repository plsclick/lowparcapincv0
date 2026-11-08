import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lowparllm.core.config import (
    AppConfig,
    DEFAULT_SYSTEM_PROMPT,
    load_config,
    persist_generation_setting,
    persist_router_setting,
)


class PhaseTwoConfigTests(unittest.TestCase):
    def test_new_config_uses_safe_defaults_and_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "lowp"
            with patch("lowparllm.core.config.config_dir", return_value=config_path):
                with patch.dict(os.environ, {}, clear=True):
                    config = load_config()

            self.assertEqual(config.base_model, "qwen3.5:4b")
            self.assertEqual(config.ollama_url, "http://127.0.0.1:11434")
            self.assertEqual(config.context_size, 4096)
            self.assertEqual(config.num_predict, -1)
            self.assertEqual(config.system_prompt, DEFAULT_SYSTEM_PROMPT)
            self.assertEqual(
                stat.S_IMODE((config_path / "config.toml").stat().st_mode), 0o600
            )

    def test_existing_user_config_is_preserved_and_missing_settings_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "lowp"
            config_path.mkdir()
            contents = '[model]\nname = "qwen3.5:4b"\n'
            (config_path / "config.toml").write_text(contents, encoding="utf-8")
            with patch("lowparllm.core.config.config_dir", return_value=config_path):
                with patch.dict(os.environ, {}, clear=True):
                    config = load_config()

            self.assertEqual(config.context_size, 4096)
            self.assertEqual(config.num_predict, -1)
            self.assertEqual(
                (config_path / "config.toml").read_text(encoding="utf-8"), contents
            )

    def test_invalid_context_and_blank_prompt_are_rejected(self) -> None:
        for environ in (
            {"LOWP_CONTEXT_SIZE": "128"},
            {"LOWP_SYSTEM_PROMPT": "  "},
        ):
            with self.subTest(environ=environ), self.assertRaises(ValueError):
                load_config(environ=environ)

    def test_unlimited_web_research_setting_can_be_enabled_from_environment(self) -> None:
        config = AppConfig.from_env(environ={"LOWP_WEB_UNLIMITED": "true"})
        self.assertTrue(config.web.unlimited)

    def test_rag_defaults_and_configurable_embedding_model(self) -> None:
        config = load_config(environ={})
        self.assertEqual(config.rag.embedding_model, "nomic-embed-text")
        self.assertEqual(config.rag.embedding_batch_size, 16)
        self.assertEqual(config.rag.chunk_size, 800)
        self.assertEqual(config.rag.chunk_overlap, 120)
        self.assertEqual(config.rag.top_k, 5)
        self.assertEqual(config.rag.score_threshold, 0.35)
        self.assertEqual(
            config.rag.qdrant_path.name,
            "qdrant",
        )
        self.assertEqual(
            config.rag.sqlite_path.name,
            "knowledge.db",
        )

        configured = AppConfig.from_env(
            environ={},
            file_config={
                "rag": {
                    "embedding": {"model": "qwen3-embedding:0.6b"},
                    "chunking": {"chunk_size": 500, "chunk_overlap": 80},
                }
            },
        )
        self.assertEqual(configured.rag.embedding_model, "qwen3-embedding:0.6b")
        self.assertEqual(configured.rag.chunk_size, 500)
        self.assertEqual(configured.rag.chunk_overlap, 80)

    def test_rag_rejects_invalid_values_and_overlapping_storage_paths(self) -> None:
        invalid_configs = (
            {"rag": {"embedding": {"batch_size": 0}}},
            {"rag": {"chunking": {"chunk_size": 100, "chunk_overlap": 100}}},
            {"rag": {"retrieval": {"score_threshold": float("nan")}}},
            {
                "rag": {
                    "storage": {
                        "qdrant_path": "/tmp/lowp-qdrant",
                        "sqlite_path": "/tmp/lowp-qdrant/knowledge.db",
                    }
                }
            },
        )
        for file_config in invalid_configs:
            with self.subTest(file_config=file_config), self.assertRaises(ValueError):
                AppConfig.from_env(environ={}, file_config=file_config)

    def test_router_settings_persist_without_rewriting_other_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "lowp"
            config_path.mkdir()
            original = (
                '[model]\nname = "qwen3.5:4b"\n\n'
                "[router]\nautomatic = true # user's preference\n"
                "max_plan_steps = 8\n"
            )
            (config_path / "config.toml").write_text(original, encoding="utf-8")

            with patch("lowparllm.core.config.config_dir", return_value=config_path):
                persist_router_setting("automatic", False)
                config = load_config()

            updated = (config_path / "config.toml").read_text(encoding="utf-8")
            self.assertFalse(config.router.automatic)
            self.assertIn('# user\'s preference', updated)
            self.assertIn("max_plan_steps = 8", updated)

    def test_generation_settings_persist_in_their_toml_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "lowp"
            config_path.mkdir()
            original = (
                '[model]\nname = "qwen3.5:4b"\nkeep_alive = "5m"\n\n'
                "[generation]\ntemperature = 0.7\ntop_p = 0.8\ntop_k = 20\n"
            )
            (config_path / "config.toml").write_text(original, encoding="utf-8")

            with patch("lowparllm.core.config.config_dir", return_value=config_path):
                persist_generation_setting("temperature", 1.25)
                persist_generation_setting("top_p", 0.95)
                persist_generation_setting("top_k", 40)
                persist_generation_setting("keep_alive", "10m")
                config = load_config()

            self.assertEqual(config.temperature, 1.25)
            self.assertEqual(config.top_p, 0.95)
            self.assertEqual(config.top_k, 40)
            self.assertEqual(config.keep_alive, "10m")

    def test_generation_setting_persistence_rejects_invalid_values(self) -> None:
        for name, value in (
            ("temperature", 2.1),
            ("top_p", float("nan")),
            ("top_k", 0),
            ("top_k", 1.5),
            ("keep_alive", " "),
            ("unknown", 1),
        ):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                persist_generation_setting(name, value)


if __name__ == "__main__":
    unittest.main()
