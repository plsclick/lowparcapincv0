"""Application configuration loaded from XDG TOML and LOWP_* overrides."""

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import tempfile
import tomllib
from typing import Mapping
from urllib.parse import urlparse

from lowparllm.core.paths import config_dir, data_dir

CONVERSATIONAL_BASE_MODEL = "qwen3.5:4b"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful local AI assistant. Be accurate, explain concepts clearly, "
    "follow the user's instructions, avoid unnecessary verbosity, admit uncertainty, "
    "use readable formatting, and maintain conversation context."
)
CONFIG_FILE = "config.toml"
HARD_MAX_WEB_RESPONSE_BYTES = 10 * 1024 * 1024
HARD_MAX_RESEARCH_CONTEXT_CHARS = 262144
_OLLAMA_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}$")
_DEFAULT_IGNORES = (
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
)


@dataclass(frozen=True)
class RagConfig:
    """Validated local document knowledge-base settings."""

    enabled: bool = True
    automatic_retrieval: bool = False
    collection_name: str = "lowp_knowledge"
    embedding_model: str = "nomic-embed-text"
    embedding_batch_size: int = 16
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    score_threshold: float = 0.35
    max_context_tokens: int = 3000
    qdrant_path: Path = field(default_factory=lambda: data_dir() / "qdrant")
    sqlite_path: Path = field(default_factory=lambda: data_dir() / "knowledge.db")
    max_file_size_mb: int = 50
    recursive: bool = True
    ignore_hidden: bool = True
    exclude_patterns: tuple[str, ...] = _DEFAULT_IGNORES


@dataclass(frozen=True)
class WebConfig:
    """Validated, conservative settings for optional internet research."""

    enabled: bool = True
    automatic_research: bool = False
    searxng_base_url: str = "http://127.0.0.1:8080"
    default_language: str = ""
    search_categories: tuple[str, ...] = ()
    safe_search: int = 1
    max_queries: int = 3
    max_results: int = 5
    max_pages: int = 5
    timeout_seconds: float = 20.0
    max_concurrent_requests: int = 3
    max_page_bytes: int = 2 * 1024 * 1024
    max_page_chars: int = 12000
    max_context_chars: int = 18000
    save_sources_enabled: bool = True
    unlimited: bool = False
    search_cache_ttl_seconds: int = 900


@dataclass
class RouterConfig:
    """Conservative routing controls; unsupported model/adapter routing stays off."""

    automatic: bool = True
    automatic_model_selection: bool = False
    automatic_adapter_selection: bool = False
    confidence_threshold: float = 0.75
    max_plan_steps: int = 8
    max_execution_seconds: int = 300
    max_concurrent_tasks: int = 2


def _section(data: Mapping[str, object], name: str) -> Mapping[str, object]:
    value: object = data
    for part in name.split("."):
        if not isinstance(value, dict):
            raise ValueError(f"Configuration section [{name}] must be a table.")
        value = value.get(part, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section [{name}] must be a table.")
    return value


def _searxng_url(value: str) -> str:
    from lowparllm.ai.search.searxng_provider import normalize_searxng_url

    return normalize_searxng_url(value)


def _rag_value(data: Mapping[str, object], section: str, field_name: str, default: object) -> object:
    return _section(data, section).get(field_name, default)


def _config_bool(
    env: Mapping[str, str], key: str, data: Mapping[str, object], field_name: str, default: bool
) -> bool:
    raw: object = env.get(key, _rag_value(data, "rag", field_name, default))
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.lower() in {"true", "1", "yes", "on"}:
        return True
    if isinstance(raw, str) and raw.lower() in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean.")


def _boolean_value(raw: object, key: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.lower() in {"true", "1", "yes", "on"}:
        return True
    if isinstance(raw, str) and raw.lower() in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean.")


def _router_number(
    env: Mapping[str, str],
    values: Mapping[str, object],
    name: str,
    env_name: str,
    default: int | float,
    minimum: int | float,
    maximum: int | float,
) -> int | float:
    raw: object = env.get(env_name, values.get(name, default))
    try:
        value: int | float = (
            float(raw) if isinstance(default, float) else int(raw)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} has an invalid value.") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{env_name} must be between {minimum} and {maximum}.")
    return value


def _rag_int(
    env: Mapping[str, str],
    env_name: str,
    data: Mapping[str, object],
    section: str,
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = env.get(env_name, str(_rag_value(data, section, name, default)))
    try:
        result = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be an integer.") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{env_name} must be between {minimum} and {maximum}.")
    return result


def _path_value(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path.")
    return path


def _positive_int(name: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return parsed


def validate_model_tag(name: str) -> str:
    """Validate a model tag accepted by LowP's Ollama model commands."""
    normalized = name.strip() if isinstance(name, str) else ""
    if (
        not _OLLAMA_MODEL_NAME.fullmatch(normalized)
        or ".." in normalized.split("/")
        or normalized.endswith(("/", ":"))
    ):
        raise ValueError("Model name must be a valid Ollama model tag.")
    return normalized


def _float_in_range(name: str, value: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return parsed


def _default_config_text() -> str:
    return (
        "[model]\n"
        f'name = "{CONVERSATIONAL_BASE_MODEL}"\n'
        'provider = "ollama"\n'
        'base_url = "http://127.0.0.1:11434"\n'
        'keep_alive = "5m"\n\n'
        "[generation]\n"
        "temperature = 0.7\n"
        "top_p = 0.8\n"
        "top_k = 20\n"
        "num_ctx = 4096\n"
        "num_predict = -1\n\n"
        "[conversation]\n"
        "max_history_messages = 40\n\n"
        "[assistant]\n"
        f"system_prompt = {json.dumps(DEFAULT_SYSTEM_PROMPT)}\n"
        "\n[rag]\n"
        "enabled = true\n"
        "automatic_retrieval = false\n"
        'collection_name = "lowp_knowledge"\n'
        "\n[rag.embedding]\n"
        'provider = "ollama"\n'
        'model = "nomic-embed-text"\n'
        "batch_size = 16\n"
        "\n[router]\n"
        "automatic = true\n"
        "automatic_model_selection = false\n"
        "automatic_adapter_selection = false\n"
        "confidence_threshold = 0.75\n"
        "max_plan_steps = 8\n"
        "max_execution_seconds = 300\n"
        "max_concurrent_tasks = 2\n"
        "\n[rag.chunking]\n"
        "chunk_size = 800\n"
        "chunk_overlap = 120\n"
        "\n[rag.retrieval]\n"
        "top_k = 5\n"
        "score_threshold = 0.35\n"
        "max_context_tokens = 3000\n"
        "\n[rag.storage]\n"
        f"qdrant_path = {json.dumps(str(data_dir() / 'qdrant'))}\n"
        f"sqlite_path = {json.dumps(str(data_dir() / 'knowledge.db'))}\n"
        "\n[rag.ingestion]\n"
        "max_file_size_mb = 50\n"
        "recursive = true\n"
        "ignore_hidden = true\n"
        'exclude_patterns = [".git", ".venv", "venv", "node_modules", '
        '"__pycache__", "build", "dist", ".tox", ".mypy_cache", '
        '".pytest_cache", ".next", "target"]\n'
        "\n[web]\n"
        "enabled = true\n"
        "automatic_research = false\n"
        'searxng_base_url = "http://127.0.0.1:8080"\n'
        'default_language = ""\n'
        "search_categories = []\n"
        "safe_search = 1\n"
        "max_queries = 3\n"
        "max_results = 5\n"
        "max_pages = 5\n"
        "timeout_seconds = 20.0\n"
        "max_concurrent_requests = 3\n"
        "max_page_bytes = 2097152\n"
        "max_page_chars = 12000\n"
        "max_context_chars = 18000\n"
        "save_sources_enabled = true\n"
        "unlimited = false\n"
        "search_cache_ttl_seconds = 900\n"
    )


def _load_user_config() -> Mapping[str, object]:
    path = config_dir() / CONFIG_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as config_file:
                config_file.write(_default_config_text())
        with path.open("rb") as config_file:
            data = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Could not load configuration from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Configuration in {path} must be a TOML table.")
    return data


def _nested_string(
    data: Mapping[str, object], section: str, field: str, default: str
) -> str:
    values = data.get(section, {})
    if not isinstance(values, dict):
        raise ValueError(f"Configuration section [{section}] must be a table.")
    value = values.get(field, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"[{section}].{field} must be a non-empty string.")
    return value


def _nested_number(
    data: Mapping[str, object],
    section: str,
    field: str,
    default: int | float,
) -> int | float:
    values = data.get(section, {})
    if not isinstance(values, dict):
        raise ValueError(f"Configuration section [{section}] must be a table.")
    value = values.get(field, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"[{section}].{field} must be a number.")
    return value


def _nested_int(
    data: Mapping[str, object], section: str, field: str, default: int
) -> int:
    value = _nested_number(data, section, field, default)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"[{section}].{field} must be an integer.")
    return int(value)


def _web_bool(
    env: Mapping[str, str], data: Mapping[str, object], name: str, default: bool
) -> bool:
    value: object = env.get(
        f"LOWP_WEB_{name.upper()}", _section(data, "web").get(name, default)
    )
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "1", "yes", "on"}:
        return True
    if isinstance(value, str) and value.lower() in {"false", "0", "no", "off"}:
        return False
    else:
        raise ValueError(f"[web].{name} must be a boolean.")


def _web_int(
    env: Mapping[str, str],
    data: Mapping[str, object],
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw: object = env.get(
        f"LOWP_WEB_{name.upper()}",
        _section(data, "web").get(name, default),
    )
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LOWP_WEB_{name.upper()} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"LOWP_WEB_{name.upper()} must be between {minimum} and {maximum}."
        )
    return value


def _web_float(
    env: Mapping[str, str],
    data: Mapping[str, object],
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw: object = env.get(
        f"LOWP_WEB_{name.upper()}",
        _section(data, "web").get(name, default),
    )
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LOWP_WEB_{name.upper()} must be a number.") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(
            f"LOWP_WEB_{name.upper()} must be between {minimum} and {maximum}."
        )
    return value


@dataclass
class AppConfig:
    """Validated settings shared by the terminal and local inference client."""

    app_name: str = "LowParLLMCapacityIncV0"
    app_version: str = "0.1.0"
    ollama_url: str = "http://127.0.0.1:11434"
    base_model: str = CONVERSATIONAL_BASE_MODEL
    keep_alive: str = "5m"
    context_size: int = 4096
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    num_predict: int = -1
    max_history_messages: int = 40
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    log_level: str = "INFO"
    rag: RagConfig = field(default_factory=RagConfig)
    web: WebConfig = field(default_factory=WebConfig)
    router: RouterConfig = field(default_factory=RouterConfig)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        file_config: Mapping[str, object] | None = None,
    ) -> "AppConfig":
        """Load validated values from TOML defaults and LOWP_* overrides."""
        env = os.environ if environ is None else environ
        data = {} if file_config is None else file_config
        url = env.get(
            "LOWP_OLLAMA_URL",
            _nested_string(data, "model", "base_url", cls.ollama_url),
        ).rstrip("/")
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("LOWP_OLLAMA_URL must be an HTTP(S) URL.")

        base_model = env.get(
            "LOWP_BASE_MODEL",
            _nested_string(data, "model", "name", cls.base_model),
        )
        try:
            base_model = validate_model_tag(base_model)
        except ValueError as exc:
            raise ValueError("LOWP_BASE_MODEL must be a valid Ollama model tag.") from exc
        if _nested_string(data, "model", "provider", "ollama") != "ollama":
            raise ValueError("The only supported inference provider is Ollama.")

        keep_alive = env.get(
            "LOWP_KEEP_ALIVE",
            _nested_string(data, "model", "keep_alive", cls.keep_alive),
        )
        temperature = float(
            env.get(
                "LOWP_TEMPERATURE",
                str(
                    _nested_number(
                        data, "generation", "temperature", cls.temperature
                    )
                ),
            )
        )
        top_p = float(
            env.get(
                "LOWP_TOP_P",
                str(_nested_number(data, "generation", "top_p", cls.top_p)),
            )
        )
        context_size = int(
            env.get(
                "LOWP_CONTEXT_SIZE",
                str(_nested_int(data, "generation", "num_ctx", cls.context_size)),
            )
        )
        num_predict = int(
            env.get(
                "LOWP_NUM_PREDICT",
                str(_nested_int(data, "generation", "num_predict", cls.num_predict)),
            )
        )
        max_history_messages = int(
            env.get(
                "LOWP_MAX_HISTORY_MESSAGES",
                str(
                    _nested_int(
                        data,
                        "conversation",
                        "max_history_messages",
                        cls.max_history_messages,
                    )
                ),
            )
        )
        system_prompt = env.get(
            "LOWP_SYSTEM_PROMPT",
            _nested_string(data, "assistant", "system_prompt", DEFAULT_SYSTEM_PROMPT),
        )
        if not system_prompt.strip():
            raise ValueError("LOWP_SYSTEM_PROMPT must not be empty.")
        top_k = _positive_int(
            "LOWP_TOP_K",
            env.get(
                "LOWP_TOP_K",
                str(_nested_int(data, "generation", "top_k", cls.top_k)),
            ),
        )
        rag_enabled = _config_bool(env, "LOWP_RAG_ENABLED", data, "enabled", True)
        automatic_retrieval = _config_bool(
            env, "LOWP_RAG_AUTOMATIC_RETRIEVAL", data, "automatic_retrieval", False
        )
        collection_name = _rag_value(
            data, "rag", "collection_name", "lowp_knowledge"
        )
        if (
            not isinstance(collection_name, str)
            or not collection_name
            or len(collection_name) > 64
            or not re.fullmatch(r"[A-Za-z0-9_-]+", collection_name)
        ):
            raise ValueError("[rag].collection_name must use letters, numbers, _ or -.")
        embedding_model = _rag_value(
            data, "rag.embedding", "model", "nomic-embed-text"
        )
        if not isinstance(embedding_model, str) or not embedding_model.strip():
            raise ValueError("[rag.embedding].model must be a non-empty string.")
        if _rag_value(data, "rag.embedding", "provider", "ollama") != "ollama":
            raise ValueError("The only supported embedding provider is Ollama.")
        rag_batch_size = _rag_int(
            env, "LOWP_RAG_BATCH_SIZE", data, "rag.embedding", "batch_size", 16, 1, 128
        )
        chunk_size = _rag_int(
            env, "LOWP_RAG_CHUNK_SIZE", data, "rag.chunking", "chunk_size", 800, 100, 8192
        )
        chunk_overlap = _rag_int(
            env,
            "LOWP_RAG_CHUNK_OVERLAP",
            data,
            "rag.chunking",
            "chunk_overlap",
            120,
            0,
            8191,
        )
        if chunk_overlap >= chunk_size:
            raise ValueError("[rag.chunking].chunk_overlap must be smaller than chunk_size.")
        retrieval_top_k = _rag_int(
            env, "LOWP_RAG_TOP_K", data, "rag.retrieval", "top_k", 5, 1, 50
        )
        threshold_raw = env.get(
            "LOWP_RAG_SCORE_THRESHOLD",
            str(_rag_value(data, "rag.retrieval", "score_threshold", 0.35)),
        )
        try:
            score_threshold = float(threshold_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("LOWP_RAG_SCORE_THRESHOLD must be a number.") from exc
        if not math.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
            raise ValueError("LOWP_RAG_SCORE_THRESHOLD must be between 0 and 1.")
        max_context_tokens = _rag_int(
            env,
            "LOWP_RAG_MAX_CONTEXT_TOKENS",
            data,
            "rag.retrieval",
            "max_context_tokens",
            3000,
            128,
            32768,
        )
        qdrant_default = data_dir() / "qdrant"
        sqlite_default = data_dir() / "knowledge.db"
        qdrant_path = _path_value(
            env.get(
                "LOWP_RAG_QDRANT_PATH",
                str(_rag_value(data, "rag.storage", "qdrant_path", str(qdrant_default))),
            ),
            "[rag.storage].qdrant_path",
        )
        sqlite_path = _path_value(
            env.get(
                "LOWP_RAG_SQLITE_PATH",
                str(_rag_value(data, "rag.storage", "sqlite_path", str(sqlite_default))),
            ),
            "[rag.storage].sqlite_path",
        )
        resolved_qdrant_path = qdrant_path.resolve()
        resolved_sqlite_path = sqlite_path.resolve()
        if (
            resolved_qdrant_path == resolved_sqlite_path
            or resolved_qdrant_path.is_relative_to(resolved_sqlite_path)
            or resolved_sqlite_path.is_relative_to(resolved_qdrant_path)
        ):
            raise ValueError(
                "The Qdrant directory and SQLite registry must not overlap."
            )
        max_file_size_mb = _rag_int(
            env,
            "LOWP_RAG_MAX_FILE_SIZE_MB",
            data,
            "rag.ingestion",
            "max_file_size_mb",
            50,
            1,
            2048,
        )
        recursive = _config_bool(
            env, "LOWP_RAG_RECURSIVE", data, "recursive", True
        )
        ignore_hidden = _config_bool(
            env, "LOWP_RAG_IGNORE_HIDDEN", data, "ignore_hidden", True
        )
        raw_patterns = _rag_value(
            data, "rag.ingestion", "exclude_patterns", list(_DEFAULT_IGNORES)
        )
        if not isinstance(raw_patterns, list) or not all(
            isinstance(pattern, str) and pattern.strip() for pattern in raw_patterns
        ):
            raise ValueError("[rag.ingestion].exclude_patterns must be an array of strings.")

        web_values = _section(data, "web")
        web_base_url = _searxng_url(
            env.get(
                "SEARXNG_BASE_URL",
                web_values.get("searxng_base_url", "http://127.0.0.1:8080"),
            )
        )
        web_language = env.get(
            "LOWP_WEB_DEFAULT_LANGUAGE",
            web_values.get("default_language", ""),
        )
        if not isinstance(web_language, str):
            raise ValueError("[web].default_language must be a string.")
        raw_categories: object = env.get(
            "LOWP_WEB_SEARCH_CATEGORIES",
            web_values.get("search_categories", []),
        )
        if isinstance(raw_categories, str):
            web_categories = tuple(
                sorted(
                    {
                        category.strip()
                        for category in raw_categories.split(",")
                        if category.strip()
                    }
                )
            )
        elif isinstance(raw_categories, list) and all(
            isinstance(category, str) and category.strip()
            for category in raw_categories
        ):
            web_categories = tuple(sorted(set(raw_categories)))
        else:
            raise ValueError("[web].search_categories must be an array of strings.")
        web_safe_search = _web_int(env, data, "safe_search", 1, 0, 2)
        web_config = WebConfig(
            enabled=_web_bool(env, data, "enabled", True),
            automatic_research=_web_bool(env, data, "automatic_research", False),
            searxng_base_url=web_base_url,
            default_language=web_language,
            search_categories=web_categories,
            safe_search=web_safe_search,
            max_queries=_web_int(env, data, "max_queries", 3, 1, 20),
            max_results=_web_int(env, data, "max_results", 10, 1, 100),
            max_pages=_web_int(env, data, "max_pages", 5, 1, 25),
            timeout_seconds=_web_float(
                env, data, "timeout_seconds", 20.0, 2.0, 60.0
            ),
            max_concurrent_requests=_web_int(
                env, data, "max_concurrent_requests", 3, 1, 8
            ),
            max_page_bytes=_web_int(
                env, data, "max_page_bytes", 2 * 1024 * 1024, 65536, 10 * 1024 * 1024
            ),
            max_page_chars=_web_int(
                env, data, "max_page_chars", 12000, 1000, 50000
            ),
            max_context_chars=_web_int(
                env, data, "max_context_chars", 18000, 2000, 60000
            ),
            save_sources_enabled=_web_bool(env, data, "save_sources_enabled", True),
            unlimited=_web_bool(env, data, "unlimited", False),
            search_cache_ttl_seconds=_web_int(
                env, data, "search_cache_ttl_seconds", 900, 0, 604800
            ),
        )
        router_values = _section(data, "router")
        router_config = RouterConfig(
            automatic=_boolean_value(
                env.get("LOWP_ROUTER_AUTOMATIC", router_values.get("automatic", True)),
                "LOWP_ROUTER_AUTOMATIC",
            ),
            automatic_model_selection=_boolean_value(
                env.get(
                    "LOWP_ROUTER_AUTO_MODEL",
                    router_values.get("automatic_model_selection", False),
                ),
                "LOWP_ROUTER_AUTO_MODEL",
            ),
            automatic_adapter_selection=_boolean_value(
                env.get(
                    "LOWP_ROUTER_AUTO_ADAPTER",
                    router_values.get("automatic_adapter_selection", False),
                ),
                "LOWP_ROUTER_AUTO_ADAPTER",
            ),
            confidence_threshold=float(_router_number(
                env, router_values, "confidence_threshold",
                "LOWP_ROUTER_CONFIDENCE_THRESHOLD", 0.75, 0.5, 1.0
            )),
            max_plan_steps=int(_router_number(
                env, router_values, "max_plan_steps",
                "LOWP_ROUTER_MAX_PLAN_STEPS", 8, 1, 16
            )),
            max_execution_seconds=int(_router_number(
                env, router_values, "max_execution_seconds",
                "LOWP_ROUTER_MAX_EXECUTION_SECONDS", 300, 10, 3600
            )),
            max_concurrent_tasks=int(_router_number(
                env, router_values, "max_concurrent_tasks",
                "LOWP_ROUTER_MAX_CONCURRENT_TASKS", 2, 1, 4
            )),
        )

        if context_size < 256 or context_size > 131072:
            raise ValueError("LOWP_CONTEXT_SIZE must be between 256 and 131072.")
        if num_predict != -1 and num_predict < 1:
            raise ValueError("LOWP_NUM_PREDICT must be -1 or a positive integer.")
        if max_history_messages < 2:
            raise ValueError("LOWP_MAX_HISTORY_MESSAGES must be at least 2.")
        if not keep_alive.strip():
            raise ValueError("LOWP_KEEP_ALIVE must not be empty.")

        log_level = env.get("LOWP_LOG_LEVEL", cls.log_level).upper()
        if log_level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("LOWP_LOG_LEVEL must be a standard logging level.")

        return cls(
            app_name=env.get("LOWP_APP_NAME", cls.app_name),
            app_version=env.get("LOWP_APP_VERSION", cls.app_version),
            ollama_url=url,
            base_model=base_model,
            keep_alive=keep_alive,
            context_size=context_size,
            temperature=_float_in_range(
                "LOWP_TEMPERATURE", str(temperature), 0.0, 2.0
            ),
            top_p=_float_in_range("LOWP_TOP_P", str(top_p), 0.0, 1.0),
            top_k=top_k,
            num_predict=num_predict,
            max_history_messages=max_history_messages,
            system_prompt=system_prompt,
            log_level=log_level,
            rag=RagConfig(
                enabled=rag_enabled,
                automatic_retrieval=automatic_retrieval,
                collection_name=collection_name,
                embedding_model=embedding_model,
                embedding_batch_size=rag_batch_size,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=retrieval_top_k,
                score_threshold=score_threshold,
                max_context_tokens=max_context_tokens,
                qdrant_path=qdrant_path,
                sqlite_path=sqlite_path,
                max_file_size_mb=max_file_size_mb,
                recursive=recursive,
                ignore_hidden=ignore_hidden,
                exclude_patterns=tuple(raw_patterns),
            ),
            web=web_config,
            router=router_config,
        )


def load_config(environ: Mapping[str, str] | None = None) -> AppConfig:
    """Load user TOML defaults, creating them exclusively if not present."""
    file_config = _load_user_config() if environ is None else {}
    return AppConfig.from_env(environ, file_config)


def persist_rag_automatic_retrieval(enabled: bool) -> None:
    """Atomically update only the RAG toggle, preserving other user config."""
    path = config_dir() / CONFIG_FILE
    if path.is_symlink():
        raise ValueError(f"Refusing to edit a symbolic-link configuration file: {path}")
    try:
        original = path.read_text(encoding="utf-8")
        mode = path.stat().st_mode & 0o777
    except OSError as exc:
        raise ValueError(f"Could not read configuration file {path}: {exc}") from exc

    replacement = f"automatic_retrieval = {'true' if enabled else 'false'}"
    lines = original.splitlines(keepends=True)
    section_start: int | None = None
    section_end = len(lines)
    for index, line in enumerate(lines):
        header = re.match(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$", line)
        if header:
            if section_start is not None:
                section_end = index
                break
            if header.group(1).strip() == "rag":
                section_start = index

    if section_start is None:
        updated = original
        if updated and not updated.endswith("\n"):
            updated += "\n"
        if updated and not updated.endswith("\n\n"):
            updated += "\n"
        updated += f"[rag]\n{replacement}\n"
    else:
        key_line: int | None = None
        for index in range(section_start + 1, section_end):
            if re.match(r"^\s*automatic_retrieval\s*=", lines[index]):
                key_line = index
                break
        if key_line is not None:
            original_line = lines[key_line]
            comment = re.search(r"\s+#.*?(?=\r?\n?$)", original_line)
            ending = "\n" if original_line.endswith("\n") else ""
            comment_text = f" {comment.group(0).lstrip()}" if comment else ""
            lines[key_line] = replacement + comment_text + ending
        else:
            lines.insert(section_start + 1, replacement + "\n")
        updated = "".join(lines)

    try:
        tomllib.loads(updated)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Refusing to write invalid TOML configuration: {exc}") from exc

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".config.toml.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(updated)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, path)
    except OSError as exc:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise ValueError(f"Could not update RAG configuration in {path}: {exc}") from exc


def persist_router_setting(name: str, value: bool | int | float) -> None:
    """Atomically persist one validated routing preference in [router]."""
    defaults: dict[str, tuple[type, int | float, int | float]] = {
        "automatic": (bool, 0, 1),
        "automatic_model_selection": (bool, 0, 1),
        "automatic_adapter_selection": (bool, 0, 1),
        "confidence_threshold": (float, 0.5, 1.0),
        "max_plan_steps": (int, 1, 16),
        "max_execution_seconds": (int, 10, 3600),
        "max_concurrent_tasks": (int, 1, 4),
    }
    if name not in defaults:
        raise ValueError(f"Unknown router setting: {name}")
    expected_type, minimum, maximum = defaults[name]
    if isinstance(value, bool):
        valid_type = expected_type is bool
    elif isinstance(value, (int, float)):
        valid_type = expected_type in {int, float} and (
            expected_type is float or isinstance(value, int)
        )
    else:
        valid_type = False
    if not valid_type or not minimum <= value <= maximum:
        raise ValueError(f"Invalid value for router setting {name!r}.")

    path = config_dir() / CONFIG_FILE
    if path.is_symlink():
        raise ValueError(f"Refusing to edit a symbolic-link configuration file: {path}")
    try:
        original = path.read_text(encoding="utf-8")
        mode = path.stat().st_mode & 0o777
    except OSError as exc:
        raise ValueError(f"Could not read configuration file {path}: {exc}") from exc
    serialized = (
        ("true" if value else "false")
        if isinstance(value, bool)
        else repr(value)
    )
    replacement = f"{name} = {serialized}"
    lines = original.splitlines(keepends=True)
    section_start: int | None = None
    section_end = len(lines)
    for index, line in enumerate(lines):
        header = re.match(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$", line)
        if not header:
            continue
        if section_start is not None:
            section_end = index
            break
        if header.group(1).strip() == "router":
            section_start = index
    if section_start is None:
        updated = original
        if updated and not updated.endswith("\n"):
            updated += "\n"
        if updated and not updated.endswith("\n\n"):
            updated += "\n"
        updated += f"[router]\n{replacement}\n"
    else:
        key_line = next(
            (
                index
                for index in range(section_start + 1, section_end)
                if re.match(rf"^\s*{re.escape(name)}\s*=", lines[index])
            ),
            None,
        )
        if key_line is None:
            lines.insert(section_start + 1, replacement + "\n")
        else:
            ending = "\n" if lines[key_line].endswith("\n") else ""
            comment = re.search(r"\s+#.*?(?=\r?\n?$)", lines[key_line])
            comment_text = f" {comment.group(0).lstrip()}" if comment else ""
            lines[key_line] = replacement + comment_text + ending
        updated = "".join(lines)
    try:
        tomllib.loads(updated)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Refusing to write invalid TOML configuration: {exc}") from exc
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".config.toml.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(updated)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, path)
    except OSError as exc:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise ValueError(f"Could not update router configuration in {path}: {exc}") from exc


def persist_generation_setting(name: str, value: str | int | float) -> None:
    """Atomically persist one validated model-generation preference."""
    settings: dict[str, tuple[str, type, int | float, int | float]] = {
        "temperature": ("generation", float, 0, 2),
        "top_p": ("generation", float, 0, 1),
        "top_k": ("generation", int, 1, float("inf")),
        "keep_alive": ("model", str, 0, 0),
    }
    if name not in settings:
        raise ValueError(f"Unknown generation setting: {name}")
    section, expected_type, minimum, maximum = settings[name]
    if expected_type is str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Keep-alive must be a non-empty duration.")
        rendered_value = json.dumps(value.strip())
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric.")
        if (
            isinstance(value, float) and not math.isfinite(value)
        ) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}.")
        if expected_type is int and not isinstance(value, int):
            raise ValueError(f"{name} must be a whole number.")
        rendered_value = repr(value)

    path = config_dir() / CONFIG_FILE
    if path.is_symlink():
        raise ValueError(f"Refusing to edit a symbolic-link configuration file: {path}")
    try:
        original = path.read_text(encoding="utf-8")
        mode = path.stat().st_mode & 0o777
    except OSError as exc:
        raise ValueError(f"Could not read configuration file {path}: {exc}") from exc

    replacement = f"{name} = {rendered_value}"
    lines = original.splitlines(keepends=True)
    section_start: int | None = None
    section_end = len(lines)
    for index, line in enumerate(lines):
        header = re.match(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$", line)
        if not header:
            continue
        if section_start is not None:
            section_end = index
            break
        if header.group(1).strip() == section:
            section_start = index

    if section_start is None:
        updated = original
        if updated and not updated.endswith("\n"):
            updated += "\n"
        if updated and not updated.endswith("\n\n"):
            updated += "\n"
        updated += f"[{section}]\n{replacement}\n"
    else:
        key_line = next(
            (
                index
                for index in range(section_start + 1, section_end)
                if re.match(rf"^\s*{re.escape(name)}\s*=", lines[index])
            ),
            None,
        )
        if key_line is None:
            lines.insert(section_start + 1, replacement + "\n")
        else:
            original_line = lines[key_line]
            comment = re.search(r"\s+#.*?(?=\r?\n?$)", original_line)
            ending = "\n" if original_line.endswith("\n") else ""
            comment_text = f" {comment.group(0).lstrip()}" if comment else ""
            lines[key_line] = replacement + comment_text + ending
        updated = "".join(lines)

    try:
        tomllib.loads(updated)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Refusing to write invalid TOML configuration: {exc}") from exc

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".config.toml.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(updated)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, path)
    except OSError as exc:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise ValueError(
            f"Could not update generation configuration in {path}: {exc}"
        ) from exc


def persist_web_setting(
    name: str, value: bool | int | float | str | tuple[str, ...]
) -> None:
    """Atomically update a supported value in [web], preserving other settings."""
    setting_ranges: dict[str, tuple[int | float, int | float]] = {
        "max_queries": (1, 20),
        "max_results": (1, 100),
        "max_pages": (1, 25),
        "timeout_seconds": (2.0, 60.0),
        "max_concurrent_requests": (1, 8),
        "max_page_bytes": (65536, 10 * 1024 * 1024),
        "max_page_chars": (1000, 50000),
        "max_context_chars": (2000, 60000),
        "safe_search": (0, 2),
        "search_cache_ttl_seconds": (0, 604800),
    }
    boolean_settings = {
        "enabled",
        "automatic_research",
        "save_sources_enabled",
        "unlimited",
    }
    string_settings = {"searxng_base_url", "default_language"}
    array_settings = {"search_categories"}
    if name not in boolean_settings | setting_ranges.keys():
        if name not in string_settings | array_settings:
            raise ValueError("Unsupported web setting.")
    if name in boolean_settings:
        if not isinstance(value, bool):
            raise ValueError("Web setting values must be boolean.")
        rendered_value = "true" if value else "false"
    elif name in string_settings:
        if not isinstance(value, str):
            raise ValueError("Web setting values must be strings.")
        rendered_value = json.dumps(value)
    elif name in array_settings:
        if not isinstance(value, tuple) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ValueError("Web search categories must be a tuple of non-empty strings.")
        rendered_value = json.dumps(list(value))
    else:
        minimum, maximum = setting_ranges[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Web setting values must be numeric.")
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be between {minimum} and {maximum}."
            )
        if name != "timeout_seconds" and not isinstance(value, int):
            raise ValueError(f"{name} must be an integer.")
        rendered_value = repr(value)

    path = config_dir() / CONFIG_FILE
    if path.is_symlink():
        raise ValueError(f"Refusing to edit a symbolic-link configuration file: {path}")
    try:
        original = path.read_text(encoding="utf-8")
        mode = path.stat().st_mode & 0o777
    except OSError as exc:
        raise ValueError(f"Could not read configuration file {path}: {exc}") from exc

    replacement = f"{name} = {rendered_value}"
    lines = original.splitlines(keepends=True)
    section_start: int | None = None
    section_end = len(lines)
    for index, line in enumerate(lines):
        header = re.match(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$", line)
        if not header:
            continue
        if section_start is not None:
            section_end = index
            break
        if header.group(1).strip() == "web":
            section_start = index

    if section_start is None:
        updated = original
        if updated and not updated.endswith("\n"):
            updated += "\n"
        if updated and not updated.endswith("\n\n"):
            updated += "\n"
        updated += f"[web]\n{replacement}\n"
    else:
        key_line = next(
            (
                index
                for index in range(section_start + 1, section_end)
                if re.match(rf"^\s*{re.escape(name)}\s*=", lines[index])
            ),
            None,
        )
        if key_line is None:
            lines.insert(section_start + 1, replacement + "\n")
        else:
            original_line = lines[key_line]
            comment = re.search(r"\s+#.*?(?=\r?\n?$)", original_line)
            ending = "\n" if original_line.endswith("\n") else ""
            comment_text = f" {comment.group(0).lstrip()}" if comment else ""
            lines[key_line] = replacement + comment_text + ending
        updated = "".join(lines)

    try:
        tomllib.loads(updated)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Refusing to write invalid TOML configuration: {exc}") from exc

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".config.toml.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(updated)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, path)
    except OSError as exc:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise ValueError(f"Could not update web configuration in {path}: {exc}") from exc
