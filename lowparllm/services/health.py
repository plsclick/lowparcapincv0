"""Non-destructive checks required for a usable LowP installation."""

from __future__ import annotations

from importlib import metadata, util
from pathlib import Path
import sqlite3
import sys
from typing import TypedDict


class HealthCheck(TypedDict):
    component: str
    status: str
    detail: str


REQUIRED_MODULES = (
    ("httpcore", "httpcore"),
    ("httpx", "httpx"),
    ("docx", "python-docx"),
    ("prompt_toolkit", "prompt-toolkit"),
    ("pypdf", "pypdf"),
    ("qdrant_client", "qdrant-client"),
    ("rich", "rich"),
)


def installation_health(config_path: Path) -> tuple[HealthCheck, ...]:
    """Check packaging, interpreter, declared runtime imports, config, and SQLite."""
    checks: list[HealthCheck] = []
    python_ok = sys.version_info >= (3, 11)
    checks.append(
        {
            "component": "Python",
            "status": "OK" if python_ok else "FAILED",
            "detail": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}; requires 3.11+",
        }
    )
    try:
        version = metadata.version("LowParLLMCapacityIncV0")
    except metadata.PackageNotFoundError:
        checks.append(
            {
                "component": "Application package",
                "status": "FAILED",
                "detail": "LowParLLMCapacityIncV0 is not installed in this interpreter.",
            }
        )
    else:
        checks.append(
            {
                "component": "Application package",
                "status": "OK",
                "detail": f"version {version}",
            }
        )
    missing = tuple(
        package
        for module, package in REQUIRED_MODULES
        if util.find_spec(module) is None
    )
    checks.append(
        {
            "component": "Required dependencies",
            "status": "FAILED" if missing else "OK",
            "detail": (
                "Missing: " + ", ".join(missing)
                if missing
                else f"{len(REQUIRED_MODULES)} required imports available"
            ),
        }
    )
    try:
        config_ok = (
            config_path.is_file()
            and not config_path.is_symlink()
            and config_path.stat().st_mode & 0o444 != 0
        )
    except OSError:
        config_ok = False
    checks.append(
        {
            "component": "Configuration",
            "status": "OK" if config_ok else "FAILED",
            "detail": str(config_path) if config_ok else f"Unreadable config file: {config_path}",
        }
    )
    try:
        with sqlite3.connect(":memory:") as connection:
            connection.execute("SELECT 1").fetchone()
    except sqlite3.Error as exc:
        checks.append(
            {"component": "SQLite", "status": "FAILED", "detail": str(exc)}
        )
    else:
        checks.append(
            {
                "component": "SQLite",
                "status": "OK",
                "detail": "SQLite runtime is operational",
            }
        )
    return tuple(checks)
