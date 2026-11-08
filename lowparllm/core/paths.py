"""Platform-appropriate user directories for LowParLLM data."""

import os
from pathlib import Path


def _is_windows() -> bool:
    return os.name == "nt"


def _home_dir() -> Path:
    if _is_windows():
        profile = os.environ.get("USERPROFILE")
        if profile:
            path = Path(profile).expanduser()
            if path.is_absolute():
                return path
    return Path.home()


def _user_directory(kind: str) -> Path:
    variable = {
        "config": "XDG_CONFIG_HOME",
        "data": "XDG_DATA_HOME",
        "cache": "XDG_CACHE_HOME",
    }[kind]
    value = os.environ.get(variable, "")
    if value:
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate / "lowp"

    home = _home_dir()
    if _is_windows():
        base_variable = "APPDATA" if kind == "config" else "LOCALAPPDATA"
        base = os.environ.get(base_variable, "")
        base_path = Path(base).expanduser() if base else Path()
        if not base_path.is_absolute():
            fallback = (
                ("AppData", "Roaming")
                if kind == "config"
                else ("AppData", "Local")
            )
            base_path = home.joinpath(*fallback)
        root = base_path / "lowp"
        return root / "cache" if kind == "cache" else root

    fallback = {
        "config": "~/.config",
        "data": "~/.local/share",
        "cache": "~/.cache",
    }[kind]
    return Path(fallback).expanduser() / "lowp"


def config_dir() -> Path:
    """Return the user configuration directory without creating it."""
    return _user_directory("config")


def data_dir() -> Path:
    """Return the user persistent-data directory without creating it."""
    return _user_directory("data")


def cache_dir() -> Path:
    """Return the user cache directory without creating it."""
    return _user_directory("cache")


def ensure_config_dir() -> Path:
    """Create and return the application configuration directory."""
    path = config_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_data_dir() -> Path:
    """Create and return the application persistent-data directory."""
    path = data_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_cache_dir() -> Path:
    """Create and return the application cache directory."""
    path = cache_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path
