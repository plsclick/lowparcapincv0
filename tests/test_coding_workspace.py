import subprocess
import shutil
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

from lowparllm.ai.coding.command_runner import DevelopmentCommandRunner, validate_command
from lowparllm.ai.coding.workspace import (
    CodingWorkspace,
    PermissionDenied,
    PermissionMode,
    WorkspaceError,
)


class CodingWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "project with spaces"
        self.root.mkdir()
        self.workspace = CodingWorkspace(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_navigation_tracks_previous_and_resolves_relative_and_home_paths(self) -> None:
        child = self.root / "sub dir"
        child.mkdir()
        self.workspace.change_directory("sub dir")
        self.assertEqual(self.workspace.cwd, child)
        self.assertEqual(self.workspace.change_directory("-"), self.root)
        self.workspace.change_directory(str(child))
        self.assertEqual(self.workspace.cwd, child)

    def test_failed_navigation_preserves_active_directory(self) -> None:
        before = self.workspace.cwd
        with self.assertRaises(WorkspaceError):
            self.workspace.change_directory("missing")
        self.assertEqual(self.workspace.cwd, before)

    def test_path_escape_and_outside_symlink_are_rejected(self) -> None:
        outside = Path(self.temporary.name) / "outside.txt"
        outside.write_text("private", encoding="utf-8")
        with self.assertRaises(WorkspaceError):
            self.workspace.read_file("../outside.txt")
        try:
            (self.root / "link.txt").symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation is not permitted in this environment")
        with self.assertRaises(WorkspaceError):
            self.workspace.read_file("link.txt")
        with self.assertRaises(WorkspaceError):
            self.workspace.resolve_file("link.txt")

    @unittest.skipUnless(shutil.which("git"), "Git is optional")
    def test_project_search_respects_gitignore(self) -> None:
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
        (self.root / "ignored.py").write_text("needle\n", encoding="utf-8")
        (self.root / "visible.py").write_text("needle\n", encoding="utf-8")

        self.assertEqual(self.workspace.find("*.py"), (self.root / "visible.py",))
        matches = self.workspace.search_text("needle")
        self.assertEqual(len(matches), 1)
        self.assertIn("visible.py:1", matches[0])

    def test_lists_hidden_files_only_when_requested_and_finds_nested_files(self) -> None:
        (self.root / ".hidden").write_text("hidden", encoding="utf-8")
        (self.root / "src").mkdir()
        (self.root / "src" / "main.py").write_text("needle", encoding="utf-8")
        self.assertNotIn(".hidden", self.workspace.list_directory())
        self.assertIn(".hidden", self.workspace.list_directory(include_hidden=True))
        self.assertEqual(self.workspace.find("*.py"), (self.root / "src" / "main.py",))
        self.assertTrue(self.workspace.search_text("needle")[0].endswith("main.py:1: needle"))

    def test_secret_files_are_not_displayed(self) -> None:
        (self.root / ".env").write_text("TOKEN=secret", encoding="utf-8")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text("[remote]\n", encoding="utf-8")
        with self.assertRaises(WorkspaceError):
            self.workspace.read_file(".env")
        with self.assertRaises(WorkspaceError):
            self.workspace.read_file(".git/config")

    def test_file_creation_and_concurrent_edit_detection(self) -> None:
        with self.assertRaises(PermissionDenied):
            self.workspace.create_file("new.py", "print('x')")
        created = self.workspace.create_file("new.py", "old\n", confirmed=True)
        self.assertEqual(created.content, "old\n")
        self.workspace.permission = PermissionMode.WORKSPACE_WRITE
        updated = self.workspace.replace_file(
            "new.py", "new\n", expected_sha256=created.sha256
        )
        self.assertEqual(updated.content, "new\n")
        with self.assertRaises(WorkspaceError):
            self.workspace.replace_file(
                "new.py", "stale\n", expected_sha256=created.sha256
            )
        self.assertEqual((self.root / "new.py").read_text(encoding="utf-8"), "new\n")
        self.assertIn("-old", self.workspace.last_diff)

    def test_delete_requires_explicit_confirmation(self) -> None:
        self.workspace.create_file("remove.txt", "remove", confirmed=True)
        with self.assertRaises(PermissionDenied):
            self.workspace.delete_file("remove.txt")
        self.workspace.delete_file("remove.txt", confirmed=True)
        self.assertFalse((self.root / "remove.txt").exists())

    def test_project_detection_reads_manifests_and_suggests_only_detected_tasks(self) -> None:
        (self.root / "pyproject.toml").write_text(
            '[project.optional-dependencies]\ndev = ["pytest"]\n'
            "[tool.ruff]\nline-length=88\n",
            encoding="utf-8",
        )
        info = self.workspace.project_info()
        self.assertIn("Python", info.languages)
        self.assertIn("pyproject.toml", info.manifests)
        self.assertEqual(
            dict(info.recommended_commands),
            {"lint": ("ruff", "check", "."), "test": ("python", "-m", "pytest")},
        )

    @unittest.skipUnless(shutil.which("git"), "Git is optional")
    def test_git_review_includes_staged_and_unstaged_diffs_without_secret_contents(self) -> None:
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(
            ["git", "-C", str(self.root), "config", "user.email", "test@example.invalid"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.root), "config", "user.name", "Test"],
            check=True,
        )
        source = self.root / "main.py"
        source.write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.root), "add", "main.py"], check=True)
        subprocess.run(["git", "-C", str(self.root), "commit", "-qm", "initial"], check=True)
        source.write_text("value = 2\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.root), "add", "main.py"], check=True)
        source.write_text("value = 3\n", encoding="utf-8")
        (self.root / ".env").write_text("TOKEN=do-not-show\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.root), "add", ".env"], check=True)
        (self.root / "new.py").write_text("new_value = True\n", encoding="utf-8")

        _, review = self.workspace.git_command("review")

        self.assertIn("Unstaged diff:", review)
        self.assertIn("Staged diff:", review)
        self.assertIn("value = 2", review)
        self.assertIn("value = 3", review)
        self.assertIn("new.py", review)
        self.assertNotIn("do-not-show", review)


class DevelopmentCommandRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_windows_commands_use_a_bounded_workspace_limited_powershell_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "notes with spaces.txt"
            source.write_text("first\nsecond\n", encoding="utf-8")
            with patch(
                "lowparllm.ai.coding.command_runner.shutil.which",
                side_effect=lambda name: "pwsh.exe" if name == "pwsh" else None,
            ):
                listing = validate_command(
                    ("Get-ChildItem",),
                    windows=True,
                    workspace_root=root,
                )
                reading = validate_command(
                    ("Get-Content", str(source), "-TotalCount", "10"),
                    windows=True,
                    workspace_root=root,
                )
                python = validate_command(
                    ("python.exe", "-m", "pytest"),
                    windows=True,
                    workspace_root=root,
                )
                verified_listing = validate_command(
                    listing,
                    windows=True,
                    workspace_root=root,
                )
                verified_reading = validate_command(
                    reading,
                    windows=True,
                    workspace_root=root,
                )
                with self.assertRaises(PermissionDenied):
                    validate_command(
                        (
                            "pwsh.exe",
                            "-NoProfile",
                            "-NonInteractive",
                            "-Command",
                            "Remove-Item -Recurse",
                        ),
                        windows=True,
                        workspace_root=root,
                    )

        self.assertEqual(listing[:3], ("pwsh.exe", "-NoProfile", "-NonInteractive"))
        self.assertIn("Select-Object -First 100", listing[-1])
        self.assertIn("-LiteralPath", reading[-1])
        self.assertIn("notes with spaces.txt", reading[-1])
        self.assertEqual(python, ("python.exe", "-m", "pytest"))
        self.assertEqual(verified_listing, listing)
        self.assertEqual(verified_reading, reading)

    async def test_windows_powershell_subset_rejects_outside_paths_and_script_injection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            root.mkdir()
            outside = Path(temporary) / "outside.txt"
            outside.write_text("private", encoding="utf-8")
            sensitive = root / ".env.local"
            sensitive.write_text("TOKEN=private", encoding="utf-8")
            with patch(
                "lowparllm.ai.coding.command_runner.shutil.which",
                side_effect=lambda name: "pwsh.exe" if name == "pwsh" else None,
            ):
                with self.assertRaises(PermissionDenied):
                    validate_command(
                        ("Get-Content", str(outside)),
                        windows=True,
                        workspace_root=root,
                    )
                with self.assertRaises(PermissionDenied):
                    validate_command(
                        ("Get-Content", str(sensitive)),
                        windows=True,
                        workspace_root=root,
                    )
                with self.assertRaises(PermissionDenied):
                    validate_command(
                        ("Get-ChildItem", ";", "Get-Process"),
                        windows=True,
                        workspace_root=root,
                    )

    async def test_commands_require_exact_confirmation_and_capture_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = CodingWorkspace(Path(temporary))
            workspace.permission = PermissionMode.WORKSPACE_WRITE
            runner = DevelopmentCommandRunner(workspace)
            argv = (sys.executable, "-m", "pytest", "--version")
            with self.assertRaises(PermissionDenied):
                await runner.run(argv)
            result = await runner.run(argv, confirmed=True)
            self.assertEqual(result.exit_code, 0)
            self.assertIn("pytest", result.stdout)
            self.assertEqual(result.cwd, temporary)

    async def test_shell_and_destructive_commands_are_blocked(self) -> None:
        for argv in (
            ("bash", "-c", "echo unsafe"),
            ("python", "-c", "print('unsafe')"),
            ("rm", "-rf", "."),
            ("git", "reset", "--hard"),
            ("git", "-c", "alias.pwn=!true", "pwn"),
            ("git", "branch", "-D", "main"),
            ("git", "diff", "--output=other.txt"),
            ("npm", "install"),
            ("pytest", "-q", ";", "rm", "-rf", "."),
        ):
            with self.subTest(argv=argv), self.assertRaises(PermissionDenied):
                validate_command(argv)

    async def test_timed_out_process_is_terminated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = CodingWorkspace(Path(temporary))
            workspace.permission = PermissionMode.WORKSPACE_WRITE
            runner = DevelopmentCommandRunner(workspace)
            result = await runner.run(
                (sys.executable, "-m", "timeit", "-n", "100000000", "-r", "1", "pass"),
                confirmed=True,
                timeout_seconds=0.1,
            )
            self.assertTrue(result.timed_out)
