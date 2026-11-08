"""Explicitly approved, shell-free development command execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import ntpath
import os
import re
import signal
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from lowparllm.ai.coding.workspace import (
    CodingWorkspace,
    PermissionDenied,
    PermissionMode,
    WorkspaceError,
)


_ALLOWED_EXECUTABLES = {
    "git",
    "pytest",
    "python",
    "python3",
    "node",
    "npm",
    "cargo",
    "go",
    "javac",
    "java",
    "php",
    "composer",
    "dart",
    "flutter",
    "make",
    "ruff",
    "mypy",
}
_BLOCKED_EXECUTABLES = {
    "bash",
    "sh",
    "zsh",
    "fish",
    "sudo",
    "su",
    "rm",
    "rmdir",
    "dd",
    "mkfs",
    "shutdown",
    "reboot",
    "poweroff",
    "chmod",
    "chown",
}
_SHELL_OPERATORS = {";", "&&", "||", "|", ">", ">>", "<", "<<", "&"}
_MAX_CAPTURE_BYTES = 500_000
_SENSITIVE_ENV_NAME = re.compile(
    r"(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.IGNORECASE
)
_DANGEROUS_GIT_ACTIONS = {"reset", "clean", "push", "rebase", "checkout", "restore"}
_ALLOWED_GIT_ACTIONS = {"status", "diff", "log", "branch", "show", "rev-parse"}


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    cwd: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    cancelled: bool = False
    output_truncated: bool = False


def validate_command(
    argv: Sequence[str],
    *,
    windows: bool | None = None,
    workspace_root: os.PathLike[str] | str | None = None,
) -> tuple[str, ...]:
    command = tuple(item for item in argv if item)
    if not command:
        raise ValueError("Provide a development command to run.")
    is_windows = os.name == "nt" if windows is None else windows
    executable = (
        ntpath.basename(command[0]).lower()
        if is_windows
        else os.path.basename(command[0]).lower()
    )
    if is_windows and executable.endswith(".exe"):
        executable = executable[:-4]
    if is_windows and executable in {"powershell", "pwsh"}:
        return _validate_windows_powershell_invocation(
            command,
            workspace_root=workspace_root,
        )
    if is_windows and executable in {
        "get-childitem",
        "get-content",
        "get-location",
        "get-process",
    }:
        return _validate_windows_powershell(
            command,
            executable,
            workspace_root=workspace_root,
        )
    if executable in _BLOCKED_EXECUTABLES:
        raise PermissionDenied(f"Command {executable!r} is blocked by LowP's safety policy.")
    if executable not in _ALLOWED_EXECUTABLES:
        raise PermissionDenied(
            f"Command {executable!r} is not in LowP's development-command allowlist."
        )
    if any(token in _SHELL_OPERATORS for token in command):
        raise PermissionDenied("Shell operators are not supported; commands run without a shell.")
    if executable in {"python", "python3"} and (
        "-c" in command or "-m" not in command and len(command) == 1
    ):
        raise PermissionDenied("Inline Python execution is blocked.")
    if executable == "git":
        if len(command) < 2 or command[1].startswith("-"):
            raise PermissionDenied("Git global options and implicit commands are not permitted.")
        if command[1] in _DANGEROUS_GIT_ACTIONS:
            raise PermissionDenied(
                f"Git {command[1]} is blocked in this phase; use Git directly after reviewing the operation."
            )
        if command[1] not in _ALLOWED_GIT_ACTIONS:
            raise PermissionDenied(f"Git {command[1]} is not an allowed read-only operation.")
        if command[1] == "branch":
            branch_args = command[2:]
            if any(
                token in {"-d", "-D", "-m", "-M", "-c", "-C", "-f", "--delete"}
                for token in branch_args
            ) or not any(
                token in {"--show-current", "--list", "-l"} for token in branch_args
            ):
                raise PermissionDenied("Only read-only Git branch listing is permitted.")
        if any(
            token in {"--output", "-o", "--ext-diff", "--textconv"}
            or token.startswith("--output=")
            for token in command[2:]
        ):
            raise PermissionDenied("Git output redirection and external diff execution are blocked.")
    if executable in {"npm", "composer"} and any(
        token in {"install", "add", "update", "publish", "exec"} for token in command[1:]
    ):
        raise PermissionDenied(
            f"{executable} install/update/network execution requires a separately reviewed manual action."
        )
    return command


class DevelopmentCommandRunner:
    """Run a bounded command only after a separate exact-command confirmation."""

    def __init__(self, workspace: CodingWorkspace) -> None:
        self.workspace = workspace

    async def run(
        self,
        argv: Sequence[str],
        *,
        confirmed: bool = False,
        timeout_seconds: float = 120,
    ) -> CommandResult:
        command = validate_command(
            argv,
            workspace_root=self.workspace.cwd,
        )
        if self.workspace.permission == PermissionMode.READ_ONLY:
            raise PermissionDenied("Command execution is disabled in read-only mode.")
        if self.workspace.permission not in PermissionMode.VALID:
            raise PermissionDenied("Unknown workspace permission mode.")
        if not confirmed:
            raise PermissionDenied(
                "Command execution requires explicit approval with `/run <command> confirm`."
            )
        if not 0.05 <= timeout_seconds <= 600:
            raise ValueError("Command timeout must be between 0.05 and 600 seconds.")

        executable = shutil.which(command[0])
        if executable is None:
            raise FileNotFoundError(f"Required command is not installed: {command[0]}")
        process = await asyncio.create_subprocess_exec(
            executable,
            *command[1:],
            cwd=self.workspace.cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                key: value
                for key, value in os.environ.items()
                if not _SENSITIVE_ENV_NAME.search(key)
            },
            **(
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                if os.name == "nt"
                else {"start_new_session": True}
            ),
        )
        stdout_task = asyncio.create_task(_read_bounded(process.stdout))
        stderr_task = asyncio.create_task(_read_bounded(process.stderr))
        timed_out = False
        cancelled = False
        try:
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
            except TimeoutError:
                timed_out = True
                await _stop_process(process)
            stdout_result, stderr_result = await asyncio.gather(stdout_task, stderr_task)
        except asyncio.CancelledError:
            cancelled = True
            await _stop_process(process)
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise

        return CommandResult(
            argv=command,
            cwd=str(self.workspace.cwd),
            exit_code=process.returncode,
            stdout=stdout_result[0].decode("utf-8", errors="replace"),
            stderr=stderr_result[0].decode("utf-8", errors="replace"),
            timed_out=timed_out,
            cancelled=cancelled,
            output_truncated=stdout_result[1] or stderr_result[1],
        )


async def _read_bounded(
    stream: asyncio.StreamReader | None,
) -> tuple[bytes, bool]:
    if stream is None:
        return b"", False
    captured = bytearray()
    truncated = False
    while chunk := await stream.read(65536):
        remaining = _MAX_CAPTURE_BYTES - len(captured)
        if remaining > 0:
            captured.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True
    return bytes(captured), truncated


def _validate_windows_powershell_invocation(
    command: tuple[str, ...],
    *,
    workspace_root: os.PathLike[str] | str | None,
) -> tuple[str, ...]:
    if len(command) != 5 or command[1:4] != (
        "-NoProfile",
        "-NonInteractive",
        "-Command",
    ):
        raise PermissionDenied(
            "Direct PowerShell scripts are blocked; use an approved read-only workspace command."
        )
    script = command[4]
    if script == "Get-Location":
        safe = _validate_windows_powershell(
            ("Get-Location",),
            "get-location",
            workspace_root=workspace_root,
        )
    elif script == "Get-Process | Select-Object -First 100":
        safe = _validate_windows_powershell(
            ("Get-Process",),
            "get-process",
            workspace_root=workspace_root,
        )
    else:
        child = re.fullmatch(
            r"Get-ChildItem -LiteralPath '((?:[^']|'')*)'((?: -(?:directory|file|force))*) \| Select-Object -First 100",
            script,
        )
        content = re.fullmatch(
            r"Get-Content -LiteralPath '((?:[^']|'')*)' -TotalCount (\d+)",
            script,
        )
        if child:
            path = child.group(1).replace("''", "'")
            options = child.group(2).split()
            safe = _validate_windows_powershell(
                ("Get-ChildItem", path, *options),
                "get-childitem",
                workspace_root=workspace_root,
            )
        elif content:
            path = content.group(1).replace("''", "'")
            safe = _validate_windows_powershell(
                ("Get-Content", path, "-TotalCount", content.group(2)),
                "get-content",
                workspace_root=workspace_root,
            )
        else:
            raise PermissionDenied(
                "Direct PowerShell scripts are blocked; use an approved read-only workspace command."
            )
    if safe[-1] != script:
        raise PermissionDenied("The PowerShell command did not pass LowP's safety checks.")
    return safe


def _validate_windows_powershell(
    command: tuple[str, ...],
    cmdlet: str,
    *,
    workspace_root: os.PathLike[str] | str | None,
) -> tuple[str, ...]:
    arguments = list(command[1:])
    root = Path(workspace_root or Path.cwd()).resolve()
    path: Path | None = None
    options: set[str] = set()

    if cmdlet in {"get-childitem", "get-content"}:
        if arguments and not arguments[0].startswith("-"):
            path = Path(arguments.pop(0))
        if cmdlet == "get-childitem":
            while arguments:
                option = arguments.pop(0).lower()
                if option not in {"-force", "-file", "-directory"} or option in options:
                    raise PermissionDenied(
                        "Get-ChildItem supports only one path and the -Force, -File, and -Directory switches."
                    )
                options.add(option)
        else:
            if arguments:
                if len(arguments) != 2 or arguments[0].lower() != "-totalcount":
                    raise PermissionDenied(
                        "Get-Content supports only a path and an optional -TotalCount from 1 to 500."
                    )
                try:
                    count = int(arguments[1])
                except ValueError as exc:
                    raise PermissionDenied("Get-Content -TotalCount must be an integer.") from exc
                if not 1 <= count <= 500:
                    raise PermissionDenied("Get-Content -TotalCount must be from 1 to 500.")
            else:
                count = 200
    elif arguments:
        raise PermissionDenied(f"{cmdlet} does not accept additional arguments.")

    resolved_path: Path | None = None
    if path is not None:
        candidate = path.expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved_path = CodingWorkspace(root).resolve_file(candidate)
        except (OSError, RuntimeError, WorkspaceError) as exc:
            raise PermissionDenied(
                "PowerShell filesystem commands are limited to existing, non-sensitive paths inside the active workspace."
            ) from exc
        if cmdlet == "get-childitem" and not resolved_path.is_dir():
            raise PermissionDenied("Get-ChildItem requires a directory inside the active workspace.")
        if cmdlet == "get-content" and not resolved_path.is_file():
            raise PermissionDenied("Get-Content requires a file inside the active workspace.")
    elif cmdlet in {"get-childitem", "get-content"}:
        resolved_path = root
        if cmdlet == "get-content":
            raise PermissionDenied("Get-Content requires a file path inside the active workspace.")

    if cmdlet == "get-childitem":
        script = "Get-ChildItem"
        if resolved_path is not None:
            literal_path = str(resolved_path).replace("'", "''")
            script += f" -LiteralPath '{literal_path}'"
        for option in sorted(options):
            script += f" {option}"
        script += " | Select-Object -First 100"
    elif cmdlet == "get-content":
        assert resolved_path is not None
        literal_path = str(resolved_path).replace("'", "''")
        script = (
            f"Get-Content -LiteralPath '{literal_path}' "
            f"-TotalCount {count}"
        )
    elif cmdlet == "get-process":
        script = "Get-Process | Select-Object -First 100"
    else:
        script = "Get-Location"

    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        raise FileNotFoundError("PowerShell is required for this Windows command.")
    return (
        powershell,
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        script,
    )


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "nt":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except (OSError, ValueError):
            process.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(process.wait(), timeout=2)
    except TimeoutError:
        if os.name == "nt":
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                result = None
            if result is None or result.returncode:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await process.wait()
