import os
from dataclasses import replace
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from lowparllm.main import main
from lowparllm.core.config import AppConfig
from lowparllm.services.health import installation_health


class InstallationHealthTests(unittest.TestCase):
    def test_health_reports_required_component_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.toml"
            config.write_text("[app]\n", encoding="utf-8")
            with (
                patch("lowparllm.services.health.metadata.version", return_value="0.1.0"),
                patch("lowparllm.services.health.util.find_spec", return_value=object()),
            ):
                checks = installation_health(config)

        self.assertTrue(checks)
        self.assertTrue(all(check["status"] in {"OK", "FAILED"} for check in checks))
        self.assertFalse(any(check["status"] == "FAILED" for check in checks))

    def test_check_mode_returns_failure_for_failed_health_check(self) -> None:
        with (
            patch.object(sys, "argv", ["lowp", "--check"]),
            patch("lowparllm.main.load_config"),
            patch("lowparllm.main.config_dir", return_value=Path("/tmp")),
            patch(
                "lowparllm.services.health.installation_health",
                return_value=(
                    {"component": "Required dependencies", "status": "FAILED", "detail": "missing"},
                ),
            ),
            patch("builtins.print") as print_message,
        ):
            self.assertEqual(main(), 1)
        print_message.assert_called_once()
        self.assertIn("FAILED: Required dependencies", print_message.call_args.args[0])

    def test_repair_delegates_to_installer_in_original_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            installer = Path(temporary) / "install.sh"
            installer.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            with (
                patch.object(sys, "argv", ["lowp", "repair"]),
                patch.dict(os.environ, {"LOWP_SOURCE_DIR": temporary}),
                patch("lowparllm.main.shutil.which", return_value="bash"),
                patch(
                    "lowparllm.main.subprocess.run",
                    return_value=type("Result", (), {"returncode": 0})(),
                ) as run,
            ):
                self.assertEqual(main(), 0)
            run.assert_called_once_with(
                ["bash", str(installer), "--repair"], check=False
            )

    def test_windows_maintenance_uses_scoped_powershell_policy_bypass(self) -> None:
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temporary:
            installer = Path(temporary) / "install.ps1"
            installer.write_text("param([switch]$Update)\n", encoding="utf-8")
            windows_os = SimpleNamespace(
                name="nt", environ={"LOWP_SOURCE_DIR": temporary}
            )
            with (
                patch.object(sys, "argv", ["lowp", "update"]),
                patch("lowparllm.main.os", windows_os),
                patch("lowparllm.main.shutil.which", return_value="pwsh"),
                patch(
                    "lowparllm.main.subprocess.run",
                    return_value=type("Result", (), {"returncode": 0})(),
                ) as run,
            ):
                self.assertEqual(main(), 0)
            run.assert_called_once_with(
                [
                    "pwsh",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(installer),
                    "-Update",
                ],
                check=False,
            )

    def test_doctor_alias_runs_the_noninteractive_health_check(self) -> None:
        with (
            patch.object(sys, "argv", ["lowp", "doctor"]),
            patch("lowparllm.main._run_doctor", return_value=0) as doctor,
        ):
            self.assertEqual(main(), 0)
        doctor.assert_called_once_with()

    def test_noninteractive_doctor_reports_optional_services_without_failing(self) -> None:
        defaults = AppConfig()
        config = replace(
            defaults,
            web=replace(defaults.web, enabled=False),
        )
        environment = {
            "system": {
                "os": "Windows",
                "distribution_version": "test",
                "architecture": "x86_64",
                "terminal": "PowerShell",
            },
            "software": {
                "git_available": True,
                "ollama": {
                    "installed": False,
                    "service_available": False,
                    "base_model_installed": False,
                    "models": [],
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("config", "data", "cache"):
                (root / directory).mkdir()
            with (
                patch.object(sys, "argv", ["lowp", "doctor"]),
                patch("lowparllm.main.load_config", return_value=config),
                patch("lowparllm.main.config_dir", return_value=root / "config"),
                patch("lowparllm.main.data_dir", return_value=root / "data"),
                patch("lowparllm.main.cache_dir", return_value=root / "cache"),
                patch("lowparllm.services.health.installation_health", return_value=()),
                patch(
                    "lowparllm.services.system_info.gather_environment",
                    return_value=environment,
                ),
                patch("lowparllm.ai.lora.adapters.AdapterManager.list", return_value=()),
                patch("builtins.print") as print_message,
            ):
                self.assertEqual(main(), 0)
        messages = "\n".join(call.args[0] for call in print_message.call_args_list)
        self.assertIn("Not configured: Ollama", messages)
        self.assertIn("Not configured: SearXNG", messages)
        self.assertIn("Local Qdrant store", messages)

    def test_repair_fails_actionably_when_checkout_is_missing(self) -> None:
        error_output = StringIO()
        with (
            patch.object(sys, "argv", ["lowp", "repair"]),
            patch.dict(os.environ, {"LOWP_SOURCE_DIR": "/missing/lowp-checkout"}),
            patch.object(sys, "stderr", error_output),
        ):
            self.assertEqual(main(), 1)
        self.assertIn("Run ./install.sh --repair", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
