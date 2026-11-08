import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from lowparllm.ai.llm.model_manager import ModelManager, ModelManagerError, persist_model_name
from lowparllm.ai.llm.ollama_client import OllamaClient
from lowparllm.core.config import AppConfig


class ModelManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_info_and_switch_persist_selected_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_root = Path(temporary) / "config" / "lowp"
            config_root.mkdir(parents=True)
            config_path = config_root / "config.toml"
            config_path.write_text(
                '[model]\nname = "qwen3.5:4b"\nprovider = "ollama"\n',
                encoding="utf-8",
            )
            config = AppConfig()
            client = OllamaClient(config.ollama_url, config.base_model)

            def response(request: httpx.Request) -> httpx.Response:
                if request.url.path == "/api/tags":
                    payload = {
                        "models": [
                            {
                                "name": "qwen3.5:4b",
                                "digest": "abc",
                                "size": 1234,
                                "details": {
                                    "family": "qwen",
                                    "parameter_size": "4B",
                                    "quantization_level": "Q4_K_M",
                                },
                            },
                            {"name": "other:latest", "digest": "def", "size": 42},
                        ]
                    }
                elif request.url.path == "/api/show":
                    payload = {
                        "details": {
                            "family": "qwen",
                            "parameter_size": "4B",
                            "quantization_level": "Q4_K_M",
                            "format": "gguf",
                        },
                        "capabilities": ["completion"],
                        "model_info": {"qwen.context_length": 32768},
                    }
                else:
                    return httpx.Response(404)
                return httpx.Response(200, json=payload)

            with patch.dict(
                os.environ,
                {"XDG_CONFIG_HOME": str(Path(temporary) / "config")},
                clear=True,
            ):
                transport = httpx.MockTransport(response)
                http = httpx.AsyncClient(transport=transport)
                manager = ModelManager(config, client, http_client=http)
                models = await manager.list_models()
                self.assertEqual(models[0].name, "qwen3.5:4b")
                info = await manager.model_info("qwen3.5:4b")
                self.assertEqual(info.context_length, 32768)
                self.assertTrue(info.chat_compatible)
                selected = await manager.use("other:latest")
                self.assertEqual(selected.name, "other:latest")
                self.assertEqual(config.base_model, "other:latest")
                self.assertEqual(client.model, "other:latest")
                self.assertIn('name = "other:latest"', config_path.read_text(encoding="utf-8"))
                await http.aclose()
                await client.aclose()

    async def test_missing_model_does_not_change_selection_or_persist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_root = Path(temporary) / "config" / "lowp"
            config_root.mkdir(parents=True)
            config_path = config_root / "config.toml"
            original = '[model]\nname = "qwen3.5:4b"\nprovider = "ollama"\n'
            config_path.write_text(original, encoding="utf-8")
            config = AppConfig()
            client = OllamaClient(config.ollama_url, config.base_model)

            def response(_request: httpx.Request) -> httpx.Response:
                del _request
                return httpx.Response(200, json={"models": []})

            with patch.dict(
                os.environ,
                {"XDG_CONFIG_HOME": str(Path(temporary) / "config")},
                clear=True,
            ):
                http = httpx.AsyncClient(transport=httpx.MockTransport(response))
                manager = ModelManager(config, client, http_client=http)
                with self.assertRaises(ModelManagerError):
                    await manager.use("missing:latest")
                self.assertEqual(config.base_model, "qwen3.5:4b")
                self.assertEqual(config_path.read_text(encoding="utf-8"), original)
                await http.aclose()
                await client.aclose()

    async def test_pull_requires_confirmation_before_request(self) -> None:
        config = AppConfig()
        client = OllamaClient(config.ollama_url, config.base_model)

        def error_response(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(500)

        http = httpx.AsyncClient(
            transport=httpx.MockTransport(error_response)
        )
        manager = ModelManager(config, client, http_client=http)
        with self.assertRaisesRegex(ModelManagerError, "explicit confirmation"):
            await manager.pull("example:latest", confirmed=False)
        await http.aclose()
        await client.aclose()

    async def test_non_chat_model_is_not_persisted_as_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_root = Path(temporary) / "config" / "lowp"
            config_root.mkdir(parents=True)
            path = config_root / "config.toml"
            original = '[model]\nname = "qwen3.5:4b"\nprovider = "ollama"\n'
            path.write_text(original, encoding="utf-8")
            config = AppConfig()
            client = OllamaClient(config.ollama_url, config.base_model)

            def response(request: httpx.Request) -> httpx.Response:
                if request.url.path == "/api/tags":
                    payload = {"models": [{"name": "embed-only", "size": 1}]}
                else:
                    payload = {"capabilities": ["embedding"]}
                return httpx.Response(200, json=payload)

            with patch.dict(
                os.environ,
                {"XDG_CONFIG_HOME": str(Path(temporary) / "config")},
                clear=True,
            ):
                http = httpx.AsyncClient(transport=httpx.MockTransport(response))
                manager = ModelManager(config, client, http_client=http)
                with self.assertRaisesRegex(ModelManagerError, "not report Ollama chat"):
                    await manager.use("embed-only")
                self.assertEqual(config.base_model, "qwen3.5:4b")
                self.assertEqual(path.read_text(encoding="utf-8"), original)
                await http.aclose()
                await client.aclose()


class ModelPersistenceTests(unittest.TestCase):
    def test_persist_model_adds_section_without_destroying_other_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "config" / "lowp"
            root.mkdir(parents=True)
            path = root / "config.toml"
            path.write_text("[generation]\ntemperature = 0.3\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"XDG_CONFIG_HOME": str(Path(temporary) / "config")},
                clear=True,
            ):
                persist_model_name("some-model:latest")
                updated = path.read_text(encoding="utf-8")
            self.assertIn("[generation]\ntemperature = 0.3", updated)
            self.assertIn('[model]\nname = "some-model:latest"', updated)


if __name__ == "__main__":
    unittest.main()
