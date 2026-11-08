"""Explicit, bounded directory traversal for supported document types."""

import fnmatch
import os
from pathlib import Path
import shutil
import subprocess

from lowparllm.ai.rag.ingestion.base import SUPPORTED_EXTENSIONS
from lowparllm.core.paths import data_dir


_DEFAULT_IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "build",
    "dist",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".next",
    "target",
}
_SENSITIVE_NAMES = {
    ".env",
    "credentials",
    "secrets",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
}
_SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
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


def discover_documents(
    root: Path,
    *,
    recursive: bool = True,
    ignore_hidden: bool = True,
    exclude_patterns: tuple[str, ...] = (),
) -> list[Path]:
    """Find explicit supported files without traversing symlinks or LowP data."""
    selected = root.expanduser()
    if selected.is_symlink():
        raise ValueError("Symbolic-link directories are not imported.")
    selected = selected.resolve(strict=True)
    if not selected.is_dir():
        raise ValueError("The selected folder is not a directory.")
    owned_data = data_dir().expanduser().resolve()
    if selected == owned_data or selected.is_relative_to(owned_data):
        raise ValueError("LowP's own application-data directory cannot be imported.")

    ignores = _DEFAULT_IGNORED_DIRECTORIES | set(exclude_patterns)
    git_visible_files = _git_visible_files(selected)
    found: list[Path] = []
    pending = [selected]
    while pending:
        current = pending.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda entry: entry.name)
        except OSError as exc:
            raise OSError(
                f"Could not inspect directory {current.name!r}: {exc.strerror or exc}"
            ) from exc
        child_dirs: list[Path] = []
        for entry in entries:
            name = entry.name
            if ignore_hidden and name.startswith("."):
                continue
            path = Path(entry.path)
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if path.resolve() == owned_data or path.resolve().is_relative_to(owned_data):
                        continue
                    relative = path.relative_to(selected).as_posix()
                    if any(
                        name in ignores
                        or fnmatch.fnmatch(name, pattern)
                        or fnmatch.fnmatch(relative, pattern)
                        for pattern in ignores
                    ):
                        continue
                    if recursive:
                        child_dirs.append(path)
                elif (
                    entry.is_file(follow_symlinks=False)
                    and path.suffix.lower() in SUPPORTED_EXTENSIONS
                    and not _is_sensitive(path.relative_to(selected))
                    and (
                        git_visible_files is None
                        or path.resolve() in git_visible_files
                    )
                ):
                    relative = path.relative_to(selected).as_posix()
                    if any(
                        fnmatch.fnmatch(name, pattern)
                        or fnmatch.fnmatch(relative, pattern)
                        for pattern in exclude_patterns
                    ):
                        continue
                    found.append(path)
            except OSError as exc:
                raise OSError(
                    f"Could not inspect entry {path.name!r}: {exc.strerror or exc}"
                ) from exc
        pending.extend(reversed(child_dirs))
    return found


def _git_visible_files(root: Path) -> set[Path] | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(
            [
                git,
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return {
        (root / os.fsdecode(item)).resolve()
        for item in result.stdout.split(b"\0")
        if item
    }


def _is_sensitive(path: Path) -> bool:
    name = path.name.lower()
    return (
        any(part.lower() in _SENSITIVE_DIRECTORIES for part in path.parts[:-1])
        or name in _SENSITIVE_NAMES
        or name.startswith(".env.")
        or name.startswith(("credentials.", "secrets."))
        or path.suffix.lower() in _SENSITIVE_SUFFIXES
    )
