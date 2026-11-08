"""SQLite-backed cache for SearXNG result sets."""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

from lowparllm.ai.search.models import SearchResult
from lowparllm.core.paths import cache_dir


class SearchCache:
    """Keep transient search results separate from the permanent RAG registry."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = (path or cache_dir() / "web-search.sqlite3").expanduser()
        if self.path.is_symlink():
            raise ValueError(f"Refusing to use a symbolic-link search cache: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS search_cache (
                    cache_key TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS search_cache_created_at "
                "ON search_cache(created_at)"
            )
        os.chmod(self.path, 0o600)

    async def get(
        self, cache_key: str, *, ttl_seconds: int
    ) -> tuple[SearchResult, ...] | None:
        if ttl_seconds <= 0:
            return None
        return await asyncio.to_thread(self._get, cache_key, ttl_seconds)

    def _get(self, cache_key: str, ttl_seconds: int) -> tuple[SearchResult, ...] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT results_json, created_at FROM search_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            try:
                created_at = datetime.fromisoformat(row[1])
            except ValueError:
                connection.execute(
                    "DELETE FROM search_cache WHERE cache_key = ?", (cache_key,)
                )
                return None
            if datetime.now(timezone.utc) - created_at > timedelta(seconds=ttl_seconds):
                connection.execute(
                    "DELETE FROM search_cache WHERE cache_key = ?", (cache_key,)
                )
                return None
            try:
                values: Any = json.loads(row[0])
                if not isinstance(values, list):
                    raise ValueError("Invalid cached results.")
                return tuple(_result_from_json(item) for item in values)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                connection.execute(
                    "DELETE FROM search_cache WHERE cache_key = ?", (cache_key,)
                )
                return None

    async def put(
        self,
        cache_key: str,
        query: str,
        parameters: dict[str, object],
        results: tuple[SearchResult, ...],
    ) -> None:
        await asyncio.to_thread(
            self._put, cache_key, query, parameters, results
        )

    def _put(
        self,
        cache_key: str,
        query: str,
        parameters: dict[str, object],
        results: tuple[SearchResult, ...],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO search_cache
                    (cache_key, query, parameters_json, results_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    query = excluded.query,
                    parameters_json = excluded.parameters_json,
                    results_json = excluded.results_json,
                    created_at = excluded.created_at
                """,
                (
                    cache_key,
                    query,
                    json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                    json.dumps(
                        [_result_to_json(result) for result in results],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    now,
                ),
            )

    async def clear(self) -> int:
        return await asyncio.to_thread(self._clear)

    def _clear(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM search_cache")
            return cursor.rowcount

    async def stats(self, *, ttl_seconds: int) -> dict[str, object]:
        return await asyncio.to_thread(self._stats, ttl_seconds)

    def _stats(self, ttl_seconds: int) -> dict[str, object]:
        with self._connect() as connection:
            if ttl_seconds > 0:
                cutoff = (
                    datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)
                ).isoformat()
                connection.execute(
                    "DELETE FROM search_cache WHERE created_at < ?", (cutoff,)
                )
            else:
                connection.execute("DELETE FROM search_cache")
            count, oldest, newest = connection.execute(
                "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM search_cache"
            ).fetchone()
        return {
            "entries": count,
            "database_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "oldest_entry": oldest,
            "newest_entry": newest,
            "ttl_seconds": ttl_seconds,
            "path": str(self.path),
        }


def _result_to_json(result: SearchResult) -> dict[str, object]:
    return {
        "title": result.title,
        "url": result.url,
        "snippet": result.snippet,
        "domain": result.domain,
        "published_date": result.published_date,
        "score": result.score,
        "engine": result.engine,
        "category": result.category,
        "position": result.position,
        "provider_metadata": result.provider_metadata,
    }


def _result_from_json(value: Any) -> SearchResult:
    if not isinstance(value, dict):
        raise ValueError("Invalid cached result.")
    return SearchResult(
        title=_required_string(value, "title"),
        url=_required_string(value, "url"),
        snippet=_required_string(value, "snippet"),
        domain=_required_string(value, "domain"),
        published_date=_optional_string(value.get("published_date")),
        score=value.get("score") if isinstance(value.get("score"), (int, float)) else None,
        engine=_optional_string(value.get("engine")),
        category=_optional_string(value.get("category")),
        position=value.get("position") if isinstance(value.get("position"), int) else None,
        provider_metadata=(
            value["provider_metadata"]
            if isinstance(value.get("provider_metadata"), dict)
            else {}
        ),
    )


def _required_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise ValueError(f"Cached result field {key} is invalid.")
    return item


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None
