import importlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from lowparllm.core.config import AppConfig, load_config
from lowparllm.core import paths
from lowparllm.services import system_info as check_environment


class ConfigTests(unittest.TestCase):
    def test_package_import_and_default_model(self) -> None:
        modules = (
            "lowparllm",
            "lowparllm.cli",
            "lowparllm.core",
            "lowparllm.core.config",
            "lowparllm.core.paths",
            "lowparllm.core.logging",
            "lowparllm.ai",
            "lowparllm.ai.llm",
            "lowparllm.ai.rag",
            "lowparllm.ai.embeddings",
            "lowparllm.ai.lora",
            "lowparllm.ai.routing",
            "lowparllm.ai.search",
            "lowparllm.ai.coding",
            "lowparllm.services",
            "lowparllm.storage",
            "lowparllm.utils",
            "lowparllm.main",
        )
        for module in modules:
            importlib.import_module(module)
        import lowparllm

        self.assertEqual(lowparllm.__version__, "0.1.0")
        self.assertEqual(load_config({}).base_model, "qwen3.5:4b")
        self.assertEqual(AppConfig.from_env({}).ollama_url, "http://127.0.0.1:11434")

    def test_default_user_config_contains_no_deferred_media_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_root = Path(temporary) / "config"
            with patch("lowparllm.core.config.config_dir", return_value=config_root):
                load_config()
            text = (config_root / "config.toml").read_text(encoding="utf-8")
        self.assertNotIn("[image]", text)
        self.assertNotIn("[pdf]", text)

    def test_environment_configuration(self) -> None:
        config = load_config(
            {
                "LOWP_OLLAMA_URL": "http://localhost:11434/",
                "LOWP_BASE_MODEL": "qwen3.5:4b",
                "LOWP_CONTEXT_SIZE": "4096",
                "LOWP_TEMPERATURE": "0.4",
                "LOWP_TOP_P": "0.9",
                "LOWP_TOP_K": "30",
                "LOWP_LOG_LEVEL": "debug",
            }
        )
        self.assertEqual(config.ollama_url, "http://localhost:11434")
        self.assertEqual(config.context_size, 4096)
        self.assertEqual(config.temperature, 0.4)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.top_k, 30)
        self.assertEqual(config.log_level, "DEBUG")

    def test_invalid_configuration_is_reported(self) -> None:
        with self.assertRaises(ValueError):
            load_config({"LOWP_CONTEXT_SIZE": "0"})
        with self.assertRaises(ValueError):
            load_config({"LOWP_OLLAMA_URL": "file:///tmp"})
        with self.assertRaisesRegex(ValueError, "valid Ollama model tag"):
            load_config({"LOWP_BASE_MODEL": "invalid model!"})
        self.assertEqual(
            load_config({"LOWP_BASE_MODEL": "another-model"}).base_model,
            "another-model",
        )


class PathTests(unittest.TestCase):
    def test_xdg_paths_are_resolved_without_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            with patch.dict(
                os.environ,
                {
                    "HOME": temporary,
                    "XDG_CONFIG_HOME": str(home / "config"),
                    "XDG_DATA_HOME": str(home / "data"),
                    "XDG_CACHE_HOME": str(home / "cache"),
                },
                clear=True,
            ):
                self.assertEqual(paths.config_dir(), home / "config/lowp")
                self.assertEqual(paths.data_dir(), home / "data/lowp")
                self.assertEqual(paths.cache_dir(), home / "cache/lowp")
                self.assertFalse((home / "config").exists())

    def test_relative_xdg_path_uses_home_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(
                os.environ,
                {"HOME": temporary, "XDG_CONFIG_HOME": "relative"},
                clear=True,
            ):
                self.assertEqual(
                    paths.config_dir(), Path(temporary) / ".config/lowp"
                )

    def test_windows_paths_use_per_user_appdata_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "User"
            roaming = home / "AppData" / "Roaming"
            local = home / "AppData" / "Local"
            with (
                patch.object(paths, "_is_windows", return_value=True),
                patch.dict(
                    os.environ,
                    {
                        "USERPROFILE": str(home),
                        "APPDATA": str(roaming),
                        "LOCALAPPDATA": str(local),
                    },
                    clear=True,
                ),
            ):
                self.assertEqual(paths.config_dir(), roaming / "lowp")
                self.assertEqual(paths.data_dir(), local / "lowp")
                self.assertEqual(paths.cache_dir(), local / "lowp/cache")


class EnvironmentDetectionTests(unittest.TestCase):
    def test_windows_environment_detection_uses_native_runtime_facts(self) -> None:
        with (
            patch.object(check_environment.platform, "system", return_value="Windows"),
            patch.object(check_environment.platform, "release", return_value="11"),
            patch.object(check_environment.platform, "version", return_value="10.0.test"),
            patch.object(check_environment.platform, "processor", return_value="Test CPU"),
            patch.object(check_environment.os, "cpu_count", return_value=8),
            patch.dict(os.environ, {"WT_SESSION": "test-session"}, clear=True),
        ):
            self.assertEqual(
                check_environment.read_os_release(),
                {"PRETTY_NAME": "Windows 11", "VERSION_ID": "10.0.test"},
            )
            self.assertEqual(
                check_environment.cpu_information(), ("Test CPU", None, 8)
            )
            self.assertEqual(check_environment._terminal_name(), "Windows Terminal")

    def test_hardware_detection_and_report_run(self) -> None:
        with (
            patch.object(check_environment, "run_command", return_value=None),
            patch.object(check_environment, "_api_request", return_value=None),
        ):
            report = check_environment.gather_environment(Path.cwd())
        self.assertIn("system", report)
        self.assertIn("total", report["memory"])
        self.assertIn("gpus", report["gpu"])
        self.assertIn("LOWPARLLM V0 — ENVIRONMENT REPORT", check_environment.render_report(report))
        self.assertIn("qwen3.5:4b", check_environment.render_report(report))

    def test_missing_optional_utilities_are_handled(self) -> None:
        with patch.object(check_environment.shutil, "which", return_value=None):
            self.assertIsNone(check_environment.run_command(["not-installed"]))
            gpu = check_environment.gpu_information()
        self.assertFalse(gpu["nvidia_driver"])
        self.assertFalse(gpu["cuda_available"])
        self.assertEqual(gpu["gpus"], [])

    def test_gpu_fallback_reports_lspci_vendor_without_inventing_vram(self) -> None:
        pci = '00:02.0 "VGA compatible controller" "Intel Corporation" "Iris Xe Graphics"'
        with (
            patch.object(
                check_environment,
                "_successful_output",
                side_effect=[None, pci, None],
            ),
            patch.object(check_environment.shutil, "which", return_value=None),
        ):
            gpu = check_environment.gpu_information()
        self.assertEqual(gpu["gpus"][0]["vendor"], "Intel Corporation")
        self.assertEqual(gpu["gpus"][0]["name"], "Intel Corporation Iris Xe Graphics")
        self.assertEqual(gpu["gpus"][0]["vram_mib"], "Unavailable")

    def test_missing_ollama_and_service_are_reported(self) -> None:
        with (
            patch.object(check_environment.shutil, "which", return_value=None),
            patch.object(check_environment, "_api_request", return_value=None),
        ):
            info = check_environment.ollama_information()
        self.assertFalse(info["installed"])
        self.assertFalse(info["service_available"])
        self.assertFalse(info["base_model_installed"])

    def test_ollama_without_service_or_model_does_not_claim_model(self) -> None:
        successful = subprocess.CompletedProcess(["ollama", "--version"], 0, "ollama 0.1\n", "")
        failed_list = subprocess.CompletedProcess(["ollama", "list"], 1, "", "connection refused")
        with (
            patch.object(check_environment.shutil, "which", return_value="/usr/bin/ollama"),
            patch.object(check_environment, "run_command", side_effect=[successful, failed_list]),
            patch.object(check_environment, "_api_request", return_value=None),
        ):
            info = check_environment.ollama_information()
        self.assertTrue(info["installed"])
        self.assertFalse(info["service_available"])
        self.assertFalse(info["base_model_installed"])

    def test_installed_model_size_and_details_are_reported(self) -> None:
        tags = {
            "models": [
                {"name": "qwen3.5:4b", "digest": "abc", "size": 1_000_000_000}
            ]
        }
        details = {
            "details": {
                "family": "qwen",
                "parameter_size": "4B",
                "quantization_level": "Q4_K_M",
            }
        }
        with (
            patch.object(check_environment.shutil, "which", return_value="/usr/bin/ollama"),
            patch.object(
                check_environment,
                "run_command",
                side_effect=[
                    subprocess.CompletedProcess(["ollama"], 0, "ollama 0.1\n", ""),
                    subprocess.CompletedProcess(["ollama"], 1, "", ""),
                ],
            ),
            patch.object(
                check_environment,
                "_api_request",
                side_effect=[tags, details, {"models": [{"name": "qwen3.5:4b"}]}],
            ),
        ):
            info = check_environment.ollama_information()
        self.assertTrue(info["base_model_installed"])
        self.assertEqual(info["base_model"]["name"], "qwen3.5:4b")
        self.assertEqual(info["base_model"]["size"], "953.7 MiB")
        self.assertIn("4B", info["model_details"])

    def test_ollama_cli_size_excludes_modified_timestamp(self) -> None:
        with (
            patch.object(check_environment.shutil, "which", return_value="/usr/bin/ollama"),
            patch.object(
                check_environment,
                "run_command",
                side_effect=[
                    subprocess.CompletedProcess(["ollama"], 0, "ollama 0.1\n", ""),
                    subprocess.CompletedProcess(
                        ["ollama"],
                        0,
                        "NAME          ID              SIZE    MODIFIED\n"
                        "qwen3.5:4b    abc123          3.3 GB  15 hours ago\n",
                        "",
                    ),
                ],
            ),
            patch.object(
                check_environment,
                "_api_request",
                side_effect=[
                    {"models": []},
                    None,
                    {"models": [{"name": "qwen3.5:4b"}]},
                ],
            ),
        ):
            info = check_environment.ollama_information()
        self.assertTrue(info["base_model_installed"])
        self.assertEqual(info["base_model"]["size"], "3.3 GB")
        self.assertEqual(info["base_model"]["modified"], "15 hours ago")

    def test_similarly_named_model_does_not_match_exact_tag(self) -> None:
        with (
            patch.object(check_environment.shutil, "which", return_value="/usr/bin/ollama"),
            patch.object(
                check_environment,
                "run_command",
                side_effect=[
                    subprocess.CompletedProcess(["ollama"], 0, "ollama 0.1\n", ""),
                    subprocess.CompletedProcess(
                        ["ollama"],
                        0,
                        "NAME                 ID        SIZE    MODIFIED\n"
                        "qwen3.5:4b-custom    abc123    3.3 GB  yesterday\n",
                        "",
                    ),
                ],
            ),
            patch.object(
                check_environment,
                "_api_request",
                return_value={"models": []},
            ),
        ):
            info = check_environment.ollama_information()
        self.assertFalse(info["base_model_installed"])
        self.assertIsNone(info["base_model"])


if __name__ == "__main__":
    unittest.main()
