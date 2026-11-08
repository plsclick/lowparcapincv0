"""Bounded project discovery and file operations for coding sessions."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import subprocess
import tomllib


_IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    "target",
    ".next",
    "coverage",
}
_SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "credentials",
    "secrets",
}
_SENSITIVE_DIRECTORIES = {
    ".aws",
    ".config",
    ".docker",
    ".git",
    ".gnupg",
    ".kube",
    ".pki",
    ".ssh",
    "credentials",
    "secrets",
}
_SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
_MANIFESTS = {
    "pyproject.toml": "Python",
    "requirements.txt": "Python",
    "setup.py": "Python",
    "package.json": "Node.js",
    "tsconfig.json": "TypeScript",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "pom.xml": "Java",
    "build.gradle": "Java/Android",
    "build.gradle.kts": "Kotlin/Android",
    "composer.json": "PHP",
    "pubspec.yaml": "Dart/Flutter",
    "CMakeLists.txt": "C/C++",
    "Makefile": "C/C++ or Make",
}
_MAX_READ_BYTES = 1_000_000
_MAX_TEXT_SEARCH_BYTES = 1_000_000


class WorkspaceError(ValueError):
    """A project operation was invalid or outside the active workspace."""


class PermissionDenied(WorkspaceError):
    """The operation exceeds the current coding permission level."""


class PermissionMode:
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    ASK = "ask"
    VALID = {READ_ONLY, WORKSPACE_WRITE, ASK}


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    content: str
    sha256: str


@dataclass(frozen=True)
class ProjectInfo:
    root: Path
    is_git_repository: bool
    git_branch: str | None
    languages: tuple[str, ...]
    manifests: tuple[str, ...]
    instructions: tuple[Path, ...]
    overview: tuple[str, ...]
    recommended_commands: tuple[tuple[str, tuple[str, ...]], ...]


class CodingWorkspace:
    """Session-scoped workspace; directory changes explicitly reset its boundary."""

    def __init__(self, initial_directory: Path | None = None) -> None:
        start = (initial_directory or Path.cwd()).expanduser().resolve()
        if not start.is_dir():
            raise WorkspaceError(f"Initial working directory does not exist: {start}")
        self.cwd = start
        self.previous_directory: Path | None = None
        self.permission = PermissionMode.ASK
        self.coding_mode = False
        self.last_diff = ""

    def set_permission(self, mode: str) -> None:
        if mode not in PermissionMode.VALID:
            raise WorkspaceError("Permission mode must be read-only, workspace-write, or ask.")
        self.permission = mode

    def change_directory(self, path: str) -> Path:
        target = self.previous_directory if path == "-" else self.resolve(path)
        if target is None:
            raise WorkspaceError("There is no previous working directory in this session.")
        if not target.is_dir():
            raise WorkspaceError(f"Not a directory: {target}")
        if target != self.cwd:
            self.previous_directory = self.cwd
            self.cwd = target
            self.last_diff = ""
        return self.cwd

    def resolve(self, path: str | Path, *, must_exist: bool = True) -> Path:
        raw_path = Path(path).expanduser()
        target = raw_path if raw_path.is_absolute() else self.cwd / raw_path
        try:
            resolved = target.resolve(strict=must_exist)
        except (OSError, RuntimeError) as exc:
            raise WorkspaceError(f"Could not resolve path {str(path)!r}: {exc}") from exc
        if not _is_within(resolved, self.cwd):
            raise WorkspaceError(
                f"Path is outside the active workspace ({self.cwd}): {resolved}"
            )
        return resolved

    def resolve_file(self, path: str | Path, *, must_exist: bool = True) -> Path:
        raw_path = Path(path).expanduser()
        candidate = raw_path if raw_path.is_absolute() else self.cwd / raw_path
        if candidate.is_symlink():
            raise WorkspaceError(f"Refusing file operations through a symbolic link: {candidate}")
        resolved = self.resolve(path, must_exist=must_exist)
        if _is_sensitive(resolved.relative_to(self.cwd)):
            raise WorkspaceError(
                f"Refusing file operations on a potentially sensitive path: {resolved}"
            )
        return resolved

    def list_directory(
        self, path: str = ".", *, include_hidden: bool = False, long: bool = False
    ) -> tuple[str, ...]:
        directory = self.resolve(path)
        if not directory.is_dir():
            raise WorkspaceError(f"Not a directory: {directory}")
        rows: list[str] = []
        try:
            entries = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold()))
            for entry in entries:
                if not include_hidden and entry.name.startswith("."):
                    continue
                try:
                    resolved = entry.resolve(strict=True)
                    if not _is_within(resolved, self.cwd):
                        continue
                    metadata = entry.lstat()
                except (OSError, RuntimeError):
                    continue
                if long:
                    kind = "d" if stat.S_ISDIR(metadata.st_mode) else "l" if stat.S_ISLNK(metadata.st_mode) else "-"
                    size = metadata.st_size
                    rows.append(
                        f"{kind}{stat.filemode(metadata.st_mode)[1:]} "
                        f"{size:>10} {entry.name}{'/' if kind == 'd' else ''}"
                    )
                else:
                    rows.append(entry.name + ("/" if entry.is_dir() else ""))
        except OSError as exc:
            raise WorkspaceError(f"Could not list {directory}: {exc}") from exc
        return tuple(rows)

    def tree(self, path: str = ".", *, max_depth: int = 3, max_entries: int = 300) -> tuple[str, ...]:
        root = self.resolve(path)
        if not root.is_dir():
            raise WorkspaceError(f"Not a directory: {root}")
        rows = [str(root)]
        count = 0

        def walk(directory: Path, prefix: str, depth: int) -> None:
            nonlocal count
            if depth >= max_depth or count >= max_entries:
                return
            try:
                children = sorted(
                    (
                        item
                        for item in directory.iterdir()
                        if not item.name.startswith(".")
                        and not (item.is_dir() and item.name in _IGNORED_DIRECTORIES)
                    ),
                    key=lambda item: (not item.is_dir(), item.name.casefold()),
                )
            except OSError:
                rows.append(f"{prefix}└── [unreadable]")
                return
            for index, item in enumerate(children):
                if count >= max_entries:
                    rows.append(f"{prefix}└── … entry limit reached")
                    return
                try:
                    resolved = item.resolve(strict=True)
                except (OSError, RuntimeError):
                    continue
                if not _is_within(resolved, self.cwd):
                    continue
                last = index == len(children) - 1
                branch = "└── " if last else "├── "
                rows.append(f"{prefix}{branch}{item.name}{'/' if item.is_dir() else ''}")
                count += 1
                if item.is_dir() and not item.is_symlink():
                    walk(item, prefix + ("    " if last else "│   "), depth + 1)

        walk(root, "", 0)
        return tuple(rows)

    def find(self, pattern: str, *, max_results: int = 200) -> tuple[Path, ...]:
        if not pattern.strip():
            raise WorkspaceError("Find pattern must not be empty.")
        results: set[Path] = set()
        for path in self._walk_files():
            candidates = [path]
            parent = path.parent
            while parent != self.cwd and _is_within(parent, self.cwd):
                candidates.append(parent)
                parent = parent.parent
            for candidate in candidates:
                if fnmatch.fnmatchcase(candidate.name.casefold(), pattern.casefold()):
                    results.add(candidate)
                    if len(results) >= max_results:
                        return tuple(sorted(results))
        return tuple(sorted(results))

    def search_text(self, needle: str, *, max_results: int = 100) -> tuple[str, ...]:
        if not needle.strip():
            raise WorkspaceError("Search text must not be empty.")
        matches: list[str] = []
        for path in self._walk_files():
            if _is_sensitive(path.relative_to(self.cwd)):
                continue
            try:
                if path.stat().st_size > _MAX_TEXT_SEARCH_BYTES:
                    continue
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for line_number, line in enumerate(text.splitlines(), 1):
                if needle.casefold() in line.casefold():
                    matches.append(f"{path.relative_to(self.cwd)}:{line_number}: {line[:500]}")
                    if len(matches) >= max_results:
                        return tuple(matches)
        return tuple(matches)

    def read_file(self, path: str) -> FileSnapshot:
        target = self.resolve_file(path)
        if not target.is_file():
            raise WorkspaceError(f"Not a regular file: {target}")
        try:
            if target.stat().st_size > _MAX_READ_BYTES:
                raise WorkspaceError(
                    f"File exceeds the {_MAX_READ_BYTES}-byte inspection limit: {target}"
                )
            raw = target.read_bytes()
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceError(f"File is not UTF-8 text: {target}") from exc
        except OSError as exc:
            raise WorkspaceError(f"Could not read {target}: {exc}") from exc
        return FileSnapshot(target, content, hashlib.sha256(raw).hexdigest())

    def create_file(self, path: str, content: str, *, confirmed: bool = False) -> FileSnapshot:
        self._require_write(confirmed)
        target = self.resolve_file(path, must_exist=False)
        if target.exists() or target.is_symlink():
            raise WorkspaceError(f"Refusing to overwrite an existing path: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
                output.write(content)
        except FileExistsError as exc:
            raise WorkspaceError(f"Refusing to overwrite an existing path: {target}") from exc
        except OSError as exc:
            raise WorkspaceError(f"Could not create {target}: {exc}") from exc
        self.last_diff = "".join(
            unified_diff(
                [],
                content.splitlines(keepends=True),
                fromfile="/dev/null",
                tofile=str(target),
            )
        )
        return self.read_file(str(target.relative_to(self.cwd)))

    def replace_file(
        self,
        path: str,
        content: str,
        *,
        expected_sha256: str,
        confirmed: bool = False,
    ) -> FileSnapshot:
        self._require_write(confirmed)
        before = self.read_file(path)
        if before.sha256 != expected_sha256:
            raise WorkspaceError(
                f"{before.path} changed since it was read; inspect it again before writing."
            )
        encoded = content.encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{before.path.name}.lowp-", dir=before.path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary_name, stat.S_IMODE(before.path.stat().st_mode))
            if hashlib.sha256(before.path.read_bytes()).hexdigest() != expected_sha256:
                raise WorkspaceError(
                    f"{before.path} changed during the edit; the original was preserved."
                )
            os.replace(temporary_name, before.path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise
        self.last_diff = "".join(
            unified_diff(
                before.content.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=str(before.path),
                tofile=str(before.path),
            )
        )
        return self.read_file(str(before.path.relative_to(self.cwd)))

    def delete_file(self, path: str, *, confirmed: bool = False) -> Path:
        self._require_write(confirmed)
        if not confirmed:
            raise PermissionDenied("Deleting a file requires explicit confirmation.")
        target = self.resolve_file(path)
        if not target.is_file() or target.is_symlink():
            raise WorkspaceError("Only regular files can be deleted by this command.")
        target.unlink()
        return target

    def project_info(self) -> ProjectInfo:
        entries = self.list_directory(".", include_hidden=True)
        manifest_names = tuple(name.rstrip("/") for name in entries if name.rstrip("/") in _MANIFESTS)
        languages = { _MANIFESTS[name] for name in manifest_names }
        for entry in entries:
            suffix = Path(entry.rstrip("/")).suffix.lower()
            languages.update(
                {
                    ".py": {"Python"},
                    ".java": {"Java"},
                    ".js": {"JavaScript"},
                    ".jsx": {"JavaScript", "React"},
                    ".ts": {"TypeScript"},
                    ".tsx": {"TypeScript", "React"},
                    ".rs": {"Rust"},
                    ".go": {"Go"},
                    ".php": {"PHP"},
                    ".dart": {"Dart/Flutter"},
                    ".c": {"C"},
                    ".h": {"C/C++"},
                    ".cpp": {"C++"},
                    ".html": {"HTML/CSS"},
                    ".css": {"HTML/CSS"},
                    ".vue": {"Vue"},
                    ".svelte": {"Svelte"},
                }.get(suffix, set())
            )
        git = shutil.which("git")
        is_repository = False
        branch: str | None = None
        if git is not None:
            import subprocess

            try:
                result = subprocess.run(
                    [git, "-C", str(self.cwd), "rev-parse", "--show-toplevel"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                is_repository = result.returncode == 0
                if is_repository:
                    branch_result = subprocess.run(
                        [git, "-C", str(self.cwd), "branch", "--show-current"],
                        capture_output=True,
                        text=True,
                        timeout=3,
                        check=False,
                    )
                    branch = branch_result.stdout.strip() or None
            except (OSError, subprocess.TimeoutExpired):
                pass
        instruction_file = self.cwd / "AGENTS.md"
        instructions = (instruction_file,) if instruction_file.is_file() else ()
        overview = tuple(entries[:80])
        recommended_commands = self.project_commands()
        return ProjectInfo(
            root=self.cwd,
            is_git_repository=is_repository,
            git_branch=branch,
            languages=tuple(sorted(languages)),
            manifests=manifest_names,
            instructions=instructions,
            overview=overview,
            recommended_commands=recommended_commands,
        )

    def project_instructions(self) -> tuple[FileSnapshot, ...]:
        instructions: list[FileSnapshot] = []
        for path in self.project_info().instructions:
            try:
                if not _is_within(path.resolve(), self.cwd):
                    continue
                instructions.append(self.read_file(str(path.relative_to(self.cwd))))
            except (WorkspaceError, OSError, UnicodeDecodeError):
                continue
        return tuple(instructions)

    def structure(self) -> tuple[str, ...]:
        return self.tree(max_depth=3)

    def git_command(self, action: str) -> tuple[int, str]:
        commands = {
            "status": ("status", "--short", "--branch"),
            "changes": ("status", "--short"),
            "diff": ("diff", "--no-ext-diff", "--"),
            "log": ("log", "-5", "--oneline", "--decorate"),
            "branch": ("branch", "--show-current"),
            "review": ("diff", "--no-ext-diff", "--unified=4", "--"),
        }
        if action not in commands:
            raise WorkspaceError("Git action must be status, changes, diff, log, branch, or review.")
        git = shutil.which("git")
        if git is None:
            raise WorkspaceError("Git is not installed.")
        if action == "review":
            output_parts: list[str] = []
            status = subprocess.run(
                [git, "-C", str(self.cwd), "status", "--short", "--branch"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if status.returncode != 0:
                raise WorkspaceError(status.stderr.strip() or "Git status failed.")
            output_parts.append("Status:\n" + (status.stdout or "(clean)"))
            for label, cached_args in (
                ("Unstaged diff", ()),
                ("Staged diff", ("--cached",)),
            ):
                names = subprocess.run(
                    [git, "-C", str(self.cwd), "diff", *cached_args, "--name-only", "-z", "--"],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                if names.returncode != 0:
                    raise WorkspaceError(names.stderr.decode(errors="replace").strip() or "Git diff failed.")
                paths = [
                    os.fsdecode(name)
                    for name in names.stdout.split(b"\0")
                    if name and not _is_sensitive(Path(os.fsdecode(name)))
                ]
                if not paths:
                    continue
                diff = subprocess.run(
                    [
                        git,
                        "-C",
                        str(self.cwd),
                        "diff",
                        *cached_args,
                        "--no-ext-diff",
                        "--no-textconv",
                        "--unified=4",
                        "--",
                        *paths,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if diff.returncode != 0:
                    raise WorkspaceError(diff.stderr.strip() or "Git diff failed.")
                output_parts.append(f"{label}:\n{diff.stdout or '(none)'}")
            output = "\n\n".join(output_parts)
            if len(output) > 100_000:
                output = output[:100_000] + "\n[review context truncated at 100,000 characters]"
            return 0, output
        result = subprocess.run(
            [git, "-C", str(self.cwd), *commands[action]],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            raise WorkspaceError(result.stderr.strip() or "Git command failed.")
        output = result.stdout
        if action in {"diff", "review"} and len(output) > 100_000:
            output = output[:100_000] + "\n[diff truncated at 100,000 characters]"
        return result.returncode, output

    def project_commands(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        commands: dict[str, tuple[str, ...]] = {}
        package_file = self.cwd / "package.json"
        if package_file.is_file():
            try:
                package = json.loads(package_file.read_text(encoding="utf-8"))
                scripts = package.get("scripts", {}) if isinstance(package, dict) else {}
                if isinstance(scripts, dict):
                    for task in ("test", "build", "lint"):
                        if task in scripts:
                            commands[task] = ("npm", "run", task)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
        pyproject = self.cwd / "pyproject.toml"
        if pyproject.is_file():
            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                if "pytest" in data.get("project", {}).get("optional-dependencies", {}).get("dev", []):
                    commands["test"] = ("python", "-m", "pytest")
                if "ruff" in data.get("tool", {}):
                    commands["lint"] = ("ruff", "check", ".")
            except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, AttributeError):
                pass
        if (self.cwd / "pytest.ini").is_file() or (self.cwd / "tests").is_dir():
            commands.setdefault("test", ("python", "-m", "pytest"))
        if (self.cwd / "Cargo.toml").is_file():
            commands.setdefault("test", ("cargo", "test"))
            commands.setdefault("build", ("cargo", "build"))
        if (self.cwd / "go.mod").is_file():
            commands.setdefault("test", ("go", "test", "./..."))
            commands.setdefault("build", ("go", "build", "./..."))
        return tuple(sorted(commands.items()))

    def _walk_files(self):
        git = shutil.which("git")
        if git is not None:
            try:
                listing = subprocess.run(
                    [git, "-C", str(self.cwd), "ls-files", "--cached", "--others",
                     "--exclude-standard", "-z"],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
                if listing.returncode == 0:
                    for raw_path in listing.stdout.split(b"\0"):
                        if not raw_path:
                            continue
                        path = self.cwd / os.fsdecode(raw_path)
                        try:
                            if (
                                not _IGNORED_DIRECTORIES.intersection(path.relative_to(self.cwd).parts)
                                and path.is_file()
                                and not path.is_symlink()
                                and _is_within(
                                path.resolve(strict=True), self.cwd
                                )
                            ):
                                yield path
                        except (OSError, RuntimeError):
                            continue
                    return
            except (OSError, subprocess.TimeoutExpired):
                pass

        for directory, subdirectories, filenames in os.walk(self.cwd, followlinks=False):
            subdirectories[:] = [
                name
                for name in subdirectories
                if name not in _IGNORED_DIRECTORIES and not name.startswith(".")
            ]
            for filename in filenames:
                path = Path(directory) / filename
                try:
                    if path.is_symlink() or not _is_within(path.resolve(strict=True), self.cwd):
                        continue
                except (OSError, RuntimeError):
                    continue
                yield path

    def _require_write(self, confirmed: bool) -> None:
        if self.permission == PermissionMode.READ_ONLY:
            raise PermissionDenied("Workspace is read-only. Use /permissions workspace-write to enable edits.")
        if self.permission == PermissionMode.ASK and not confirmed:
            raise PermissionDenied(
                "This operation needs approval. Use the command's confirm argument or /permissions workspace-write."
            )
        if self.permission not in PermissionMode.VALID:
            raise PermissionDenied("Unknown workspace permission mode.")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_sensitive(path: Path) -> bool:
    name = path.name.lower()
    return (
        any(part.lower() in _SENSITIVE_DIRECTORIES for part in path.parts[:-1])
        or name in _SENSITIVE_NAMES
        or name.startswith(".env.")
        or name.startswith(("credentials.", "secrets."))
        or path.suffix.lower() in _SENSITIVE_SUFFIXES
    )
