import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from lowparllm import main as main_module
from lowparllm.core.config import AppConfig

ROOT = Path(__file__).resolve().parents[1]


class EntryPointTests(unittest.TestCase):
    def test_main_runs_the_terminal_application(self) -> None:
        app = Mock()
        with (
            patch.object(main_module, "load_config", return_value=AppConfig()),
            patch("lowparllm.cli.terminal.TerminalApp", return_value=app) as terminal,
            patch.object(main_module.asyncio, "run", return_value=0) as run,
            patch.object(sys, "argv", ["lowp"]),
        ):
            self.assertEqual(main_module.main(), 0)
        terminal.assert_called_once()
        run.assert_called_once()

    @unittest.skipUnless(sys.platform.startswith("linux"), "Phase 1 supports Linux only")
    def test_module_and_console_entrypoints_run_from_another_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "home"
            workdir = Path(temporary) / "outside-project"
            home.mkdir()
            workdir.mkdir()
            environment = os.environ.copy()
            environment.update(
                {
                    "HOME": str(home),
                    "XDG_CONFIG_HOME": str(home / ".config"),
                    "XDG_DATA_HOME": str(home / ".local/share"),
                    "XDG_CACHE_HOME": str(home / ".cache"),
                    "LOWP_OLLAMA_URL": "http://127.0.0.1:1",
                    "TERM": "xterm-256color",
                    "PATH": str(Path(sys.executable).parent),
                }
            )
            for command in (
                [sys.executable, "-m", "lowparllm"],
                [str(ROOT / ".venv/bin/lowp")],
            ):
                with self.subTest(command=command):
                    self._run_interactive_command(command, workdir, environment)

    def _run_interactive_command(self, command, workdir, environment) -> None:
        import pty
        import termios

        master, slave = pty.openpty()
        fcntl_window_size = struct.pack("HHHH", 24, 80, 0, 0)
        import fcntl

        fcntl.ioctl(slave, termios.TIOCSWINSZ, fcntl_window_size)
        process = subprocess.Popen(
            command,
            cwd=workdir,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=environment,
            close_fds=True,
        )
        os.close(slave)
        output = bytearray()
        deadline = time.monotonic() + 20
        sent_exit = False
        try:
            while time.monotonic() < deadline:
                ready, _, _ = select.select([master], [], [], 0.2)
                if ready:
                    try:
                        output.extend(os.read(master, 8192))
                    except OSError:
                        break
                if b"lowp " in output and b"outside-project >" in output and not sent_exit:
                    os.write(master, b"/exit\r")
                    sent_exit = True
                if process.poll() is not None:
                    break
            process.wait(timeout=max(1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            self.fail(f"terminal command timed out; output: {output.decode(errors='replace')}")
        finally:
            os.close(master)
        rendered = output.decode(errors="replace")
        self.assertEqual(process.returncode, 0, rendered)
        self.assertIn("Welcome to LowParLLMCapacityIncV0", rendered)
        self.assertIn("Goodbye.", rendered)

    def test_invalid_configuration_returns_clear_error(self) -> None:
        with (
            patch.object(main_module, "load_config", side_effect=ValueError("bad setting")),
            patch("builtins.print") as print_message,
            patch.object(sys, "argv", ["lowp"]),
        ):
            self.assertEqual(main_module.main(), 2)
        print_message.assert_called_once()
        self.assertIn("Invalid LowParLLM configuration", print_message.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
