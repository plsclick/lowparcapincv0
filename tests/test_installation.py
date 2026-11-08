import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "install.sh"
UNINSTALLER = ROOT / "uninstall.sh"
MARKER = "# Managed by LowParLLMCapacityIncV0 installer."
IS_LINUX = sys.platform.startswith("linux")
IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


class InstallationScriptTests(unittest.TestCase):
    def test_installer_uses_project_dependencies_without_media_generation(self) -> None:
        script = INSTALLER.read_text(encoding="utf-8")
        manifest = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('pip install --upgrade "$script_dir"', script)
        self.assertIn('"$launcher" --check', script)
        self.assertIn("--repair", script)
        self.assertIn("--update", script)
        self.assertIn('obsolete = ("diffusers", "reportlab", "Pillow")', script)
        self.assertIn("distribution.requires or ()", script)
        self.assertNotIn("[image]", manifest)
        self.assertNotIn("reportlab", manifest.lower())
        self.assertNotIn("diffusers", manifest.lower())
        self.assertNotIn("image-generation", script.lower())
        self.assertNotIn("huggingface.co", script)
        self.assertNotIn("hf download", script)
        self.assertIn("ollama list", script)
        self.assertIn("ollama pull", script)
        self.assertIn("Download it now with ollama pull?", script)
        self.assertIn("search?", script)
        self.assertIn('"results") if isinstance(payload, dict) else None', script)
        self.assertIn("127.0.0.1:8080:8080", script)
        self.assertIn("io.lowp.managed=true", script)
        self.assertIn("searxng/searxng@sha256:", script)
        self.assertNotIn("sudo apt", script)

    def test_compose_uses_the_same_immutable_searxng_image_as_installer(self) -> None:
        script = INSTALLER.read_text(encoding="utf-8")
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
        image = next(
            line.strip().removeprefix("image: ")
            for line in compose.splitlines()
            if line.strip().startswith("image:")
        )
        self.assertIn(f'image="{image}"', script)
        self.assertRegex(image, r"^searxng/searxng@sha256:[0-9a-f]{64}$")

    def test_windows_installation_scripts_are_user_scoped_and_non_destructive(self) -> None:
        installer = (ROOT / "install.ps1").read_text(encoding="utf-8")
        uninstaller = (ROOT / "uninstall.ps1").read_text(encoding="utf-8")
        self.assertIn('GetFolderPath("LocalApplicationData")', installer)
        self.assertIn("-m pip install --upgrade $ScriptDirectory", installer)
        self.assertIn('SetEnvironmentVariable("Path", $newUserPath, "User")', installer)
        self.assertIn("ollama list", installer)
        self.assertNotIn("ollama pull", installer.lower())
        self.assertNotIn("Set-ExecutionPolicy", installer)
        self.assertIn("Type REMOVE to confirm", uninstaller)
        self.assertIn("User data was preserved", uninstaller)
        self.assertNotIn("ollama", uninstaller.lower())
        self.assertIn("Programs\\LowP", uninstaller)

    @unittest.skipUnless(IS_LINUX, "Bash installer checks run on Linux")
    def test_shell_scripts_parse(self) -> None:
        for script in (INSTALLER, UNINSTALLER):
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["bash", "-n", str(script)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(IS_LINUX, "Linux installer lifecycle test")
    @unittest.skipIf(IS_ROOT, "installer intentionally refuses root execution")
    def test_installer_preserves_unrelated_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            launcher = home / ".local/bin/lowp"
            launcher.parent.mkdir(parents=True)
            launcher.write_text("#!/bin/sh\nprintf user-owned\n", encoding="utf-8")
            original = launcher.read_bytes()
            environment = os.environ.copy()
            environment.update(
                {
                    "HOME": str(home),
                    "XDG_DATA_HOME": str(home / ".local/share"),
                    "PYTHON": shutil.which("python3") or "python3",
                }
            )
            result = subprocess.run(
                ["bash", str(INSTALLER)],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite an unrelated file", result.stderr)
            self.assertEqual(launcher.read_bytes(), original)
            self.assertFalse((home / ".local/share/lowp/venv").exists())

    @unittest.skipUnless(IS_LINUX, "Linux installer lifecycle test")
    @unittest.skipIf(IS_ROOT, "installer intentionally refuses root execution")
    def test_installer_preserves_unrecognized_runtime_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            runtime = home / ".local/share/lowp/venv"
            runtime.mkdir(parents=True)
            marker = runtime / ".lowp-installed"
            marker.write_text("not the LowParLLM marker\n", encoding="utf-8")
            sentinel = runtime / "user-file"
            sentinel.write_text("keep", encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                {
                    "HOME": str(home),
                    "XDG_DATA_HOME": str(home / ".local/share"),
                    "PYTHON": shutil.which("python3") or "python3",
                }
            )
            result = subprocess.run(
                ["bash", str(INSTALLER)],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ownership marker is not recognized", result.stderr)
            self.assertEqual(sentinel.read_text(), "keep")

    @unittest.skipUnless(IS_LINUX, "Linux uninstaller lifecycle test")
    def test_uninstaller_decline_then_removes_only_owned_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            data_home = home / ".local/share"
            runtime = data_home / "lowp/venv"
            runtime.mkdir(parents=True)
            (runtime / ".lowp-installed").write_text(MARKER + "\n", encoding="utf-8")
            (runtime / "runtime-file").write_text("runtime", encoding="utf-8")
            user_data = data_home / "lowp/conversations"
            user_data.mkdir()
            (user_data / "keep.db").write_text("user data", encoding="utf-8")
            launcher = home / ".local/bin/lowp"
            launcher.parent.mkdir(parents=True)
            launcher.write_text(
                f"#!/usr/bin/env bash\n{MARKER}\nexit 0\n", encoding="utf-8"
            )
            developer_venv = home / "repository/.venv"
            developer_venv.mkdir(parents=True)
            (developer_venv / "keep").write_text("developer environment", encoding="utf-8")
            environment = os.environ.copy()
            environment.update({"HOME": str(home), "XDG_DATA_HOME": str(data_home)})

            cancelled = subprocess.run(
                ["bash", str(UNINSTALLER)],
                input="n\n",
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(cancelled.returncode, 0, cancelled.stderr)
            self.assertTrue(runtime.exists())
            self.assertTrue(launcher.exists())

            removed = subprocess.run(
                ["bash", str(UNINSTALLER)],
                input="y\n",
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(removed.returncode, 0, removed.stderr)
            self.assertFalse(runtime.exists())
            self.assertFalse(launcher.exists())
            self.assertEqual((user_data / "keep.db").read_text(), "user data")
            self.assertEqual(
                (developer_venv / "keep").read_text(), "developer environment"
            )
            repeated = subprocess.run(
                ["bash", str(UNINSTALLER)],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertIn("Nothing was removed", repeated.stdout)

    @unittest.skipUnless(IS_LINUX, "Linux uninstaller lifecycle test")
    def test_uninstaller_data_purge_requires_explicit_flag_and_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            data_home = home / ".local/share"
            config = home / ".config/lowp"
            user_data = data_home / "lowp"
            cache = home / ".cache/lowp"
            for directory in (config, user_data, cache):
                directory.mkdir(parents=True)
                (directory / "keep").write_text("user-owned", encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                {
                    "HOME": str(home),
                    "XDG_DATA_HOME": str(data_home),
                    "XDG_CONFIG_HOME": str(home / ".config"),
                    "XDG_CACHE_HOME": str(home / ".cache"),
                }
            )
            cancelled = subprocess.run(
                ["bash", str(UNINSTALLER), "--purge-data"],
                input="y\nn\n",
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(cancelled.returncode, 0, cancelled.stderr)
            self.assertTrue((config / "keep").exists())
            self.assertTrue((user_data / "keep").exists())
            self.assertTrue((cache / "keep").exists())

            removed = subprocess.run(
                ["bash", str(UNINSTALLER), "--purge-data"],
                input="y\ny\n",
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(removed.returncode, 0, removed.stderr)
            self.assertFalse(config.exists())
            self.assertFalse(user_data.exists())
            self.assertFalse(cache.exists())

    @unittest.skipUnless(IS_LINUX, "Linux uninstaller lifecycle test")
    def test_uninstaller_preserves_unowned_launcher_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            data_home = home / ".local/share"
            runtime = data_home / "lowp/venv"
            runtime.mkdir(parents=True)
            (runtime / ".lowp-installed").write_text(
                "user-owned marker\n", encoding="utf-8"
            )
            (runtime / "user-file").write_text("keep", encoding="utf-8")
            launcher = home / ".local/bin/lowp"
            launcher.parent.mkdir(parents=True)
            launcher.write_text("#!/bin/sh\nuser owned\n", encoding="utf-8")
            environment = os.environ.copy()
            environment.update({"HOME": str(home), "XDG_DATA_HOME": str(data_home)})
            result = subprocess.run(
                ["bash", str(UNINSTALLER)],
                input="y\n",
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((runtime / "user-file").exists())
            self.assertEqual(launcher.read_text(), "#!/bin/sh\nuser owned\n")

    @unittest.skipUnless(IS_LINUX, "Linux uninstaller lifecycle test")
    def test_uninstaller_removes_partial_marked_runtime_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            data_home = home / ".local/share"
            runtime = data_home / "lowp/venv"
            runtime.mkdir(parents=True)
            (runtime / ".lowp-installed").write_text(MARKER + "\n", encoding="utf-8")
            (runtime / "partial-file").write_text("runtime", encoding="utf-8")
            environment = os.environ.copy()
            environment.update({"HOME": str(home), "XDG_DATA_HOME": str(data_home)})

            removed = subprocess.run(
                ["bash", str(UNINSTALLER)],
                input="y\n",
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(removed.returncode, 0, removed.stderr)
            self.assertFalse(runtime.exists())

            repeated = subprocess.run(
                ["bash", str(UNINSTALLER)],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertIn("Nothing was removed", repeated.stdout)

    @unittest.skipUnless(IS_LINUX, "Linux uninstaller lifecycle test")
    def test_uninstaller_requires_separate_searxng_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            data_home = home / ".local/share"
            searxng = data_home / "lowp/searxng"
            searxng.mkdir(parents=True)
            (searxng / ".lowp-searxng-managed").write_text(
                MARKER + "\n", encoding="utf-8"
            )
            (searxng / "settings.yml").write_text("keep: true\n", encoding="utf-8")
            fake_bin = home / "bin"
            fake_bin.mkdir()
            docker_log = home / "docker.log"
            docker = fake_bin / "docker"
            docker.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "container" ] && [ "$2" = "inspect" ]; then\n'
                '  if [ "$3" = "--format" ]; then echo true; else echo "{}"; fi\n'
                "  exit 0\n"
                "fi\n"
                'if [ "$1" = "container" ] && [ "$2" = "rm" ]; then\n'
                '  printf "%s\\n" "$*" >> "$FAKE_DOCKER_LOG"\n'
                "  exit 0\n"
                "fi\n"
                "exit 1\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                {
                    "HOME": str(home),
                    "XDG_DATA_HOME": str(data_home),
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "FAKE_DOCKER_LOG": str(docker_log),
                }
            )
            result = subprocess.run(
                ["bash", str(UNINSTALLER)],
                input="y\nn\n",
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((searxng / "settings.yml").exists())
            self.assertFalse(docker_log.exists())

    @unittest.skipUnless(IS_LINUX, "Linux uninstaller lifecycle test")
    def test_uninstaller_removes_only_confirmed_owned_searxng_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            data_home = home / ".local/share"
            searxng = data_home / "lowp/searxng"
            searxng.mkdir(parents=True)
            (searxng / ".lowp-searxng-managed").write_text(
                MARKER + "\n", encoding="utf-8"
            )
            fake_bin = home / "bin"
            fake_bin.mkdir()
            docker_log = home / "docker.log"
            docker = fake_bin / "docker"
            docker.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "container" ] && [ "$2" = "inspect" ]; then\n'
                '  if [ "$3" = "--format" ]; then echo true; else echo "{}"; fi\n'
                "  exit 0\n"
                "fi\n"
                'if [ "$1" = "container" ] && [ "$2" = "rm" ]; then\n'
                '  printf "%s\\n" "$*" >> "$FAKE_DOCKER_LOG"\n'
                "  exit 0\n"
                "fi\n"
                "exit 1\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                {
                    "HOME": str(home),
                    "XDG_DATA_HOME": str(data_home),
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "FAKE_DOCKER_LOG": str(docker_log),
                }
            )
            result = subprocess.run(
                ["bash", str(UNINSTALLER)],
                input="y\ny\n",
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(searxng.exists())
            removals = docker_log.read_text()
            self.assertIn("container rm --force lowp-searxng", removals)
            self.assertIn("container rm --force lowp-searxng-previous", removals)


if __name__ == "__main__":
    unittest.main()
