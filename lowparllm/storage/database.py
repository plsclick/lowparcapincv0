"""SQLite document registry and non-vector knowledge metadata."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Sequence


@dataclass(frozen=True)
class SourceRecord:
    id: str
    filename: str
    original_path: str
    file_type: str
    file_size: int
    content_hash: str
    imported_at: str
    indexed_at: str | None
    status: str
    chunk_count: int
    error: str | None
    embedding_model: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    chunk_index: int
    content_hash: str
    page_number: int | None
    section_heading: str | None
    start_line: int
    end_line: int


class DocumentRegistry:
    """Own one SQLite connection and commit metadata changes transactionally."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        if self.path.is_symlink():
            raise ValueError(f"Refusing to use a symbolic-link registry file: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.path, timeout=10, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._initialize()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                original_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                indexed_at TEXT,
                status TEXT NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                embedding_model TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE UNIQUE INDEX IF NOT EXISTS documents_content_hash
                ON documents(content_hash) WHERE status = 'indexed';
            CREATE INDEX IF NOT EXISTS documents_path ON documents(original_path);
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                page_number INTEGER,
                section_heading TEXT,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                UNIQUE(document_id, chunk_index)
            );
            CREATE INDEX IF NOT EXISTS chunks_document_id ON chunks(document_id);
            CREATE TABLE IF NOT EXISTS rag_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

    def close(self) -> None:
        self._connection.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _source(self, row: sqlite3.Row) -> SourceRecord:
        return SourceRecord(
            id=row["id"],
            filename=row["filename"],
            original_path=row["original_path"],
            file_type=row["file_type"],
            file_size=row["file_size"],
            content_hash=row["content_hash"],
            imported_at=row["imported_at"],
            indexed_at=row["indexed_at"],
            status=row["status"],
            chunk_count=row["chunk_count"],
            error=row["error"],
            embedding_model=row["embedding_model"],
            metadata=json.loads(row["metadata_json"]),
        )

    def get_source(self, source_id: str) -> SourceRecord | None:
        row = self._connection.execute(
            "SELECT * FROM documents WHERE id = ?", (source_id,)
        ).fetchone()
        if row is not None:
            return self._source(row)
        matches = self._connection.execute(
            "SELECT * FROM documents WHERE id LIKE ? ORDER BY id LIMIT 2",
            (f"{source_id}%",),
        ).fetchall()
        return self._source(matches[0]) if len(matches) == 1 else None

    def find_hash(self, content_hash: str) -> SourceRecord | None:
        row = self._connection.execute(
            "SELECT * FROM documents WHERE content_hash = ? AND status = 'indexed'",
            (content_hash,),
        ).fetchone()
        return self._source(row) if row is not None else None

    def find_path(self, path: str) -> SourceRecord | None:
        row = self._connection.execute(
            "SELECT * FROM documents WHERE original_path = ? ORDER BY imported_at DESC LIMIT 1",
            (path,),
        ).fetchone()
        return self._source(row) if row is not None else None

    def update_path(self, source_id: str, path: Path, size: int) -> SourceRecord:
        with self._connection:
            self._connection.execute(
                "UPDATE documents SET original_path = ?, filename = ?, file_size = ?, "
                "error = NULL WHERE id = ?",
                (str(path), path.name, size, source_id),
            )
        source = self.get_source(source_id)
        if source is None:
            raise RuntimeError("The indexed source disappeared during path update.")
        return source

    def register_indexed(
        self,
        *,
        source_id: str,
        path: Path,
        file_type: str,
        file_size: int,
        file_hash: str,
        chunks: Sequence[ChunkRecord],
        embedding_model: str,
        metadata: dict[str, Any],
    ) -> None:
        now = self._now()
        existing = self.get_source(source_id)
        imported_at = existing.imported_at if existing else now
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO documents (
                    id, filename, original_path, file_type, file_size, content_hash,
                    imported_at, indexed_at, status, chunk_count, error,
                    embedding_model, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'indexed', ?, NULL, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    filename = excluded.filename,
                    original_path = excluded.original_path,
                    file_type = excluded.file_type,
                    file_size = excluded.file_size,
                    content_hash = excluded.content_hash,
                    indexed_at = excluded.indexed_at,
                    status = 'indexed',
                    chunk_count = excluded.chunk_count,
                    error = NULL,
                    embedding_model = excluded.embedding_model,
                    metadata_json = excluded.metadata_json
                """,
                (
                    source_id,
                    path.name,
                    str(path),
                    file_type,
                    file_size,
                    file_hash,
                    imported_at,
                    now,
                    len(chunks),
                    embedding_model,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            self._connection.execute(
                "DELETE FROM chunks WHERE document_id = ?", (source_id,)
            )
            self._connection.executemany(
                """
                INSERT INTO chunks (
                    id, document_id, chunk_index, content_hash, page_number,
                    section_heading, start_line, end_line
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        source_id,
                        chunk.chunk_index,
                        chunk.content_hash,
                        chunk.page_number,
                        chunk.section_heading,
                        chunk.start_line,
                        chunk.end_line,
                    )
                    for chunk in chunks
                ],
            )
            self._connection.executemany(
                "INSERT INTO rag_metadata(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (
                    ("embedding_model", embedding_model),
                    ("vector_dimensions", str(metadata.get("vector_dimensions", ""))),
                ),
            )

    def record_failure(
        self,
        *,
        source_id: str,
        path: Path,
        file_type: str,
        file_size: int,
        file_hash: str,
        error: str,
    ) -> None:
        existing = self.get_source(source_id)
        status = "stale" if existing and existing.status == "indexed" else "failed"
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO documents (
                    id, filename, original_path, file_type, file_size, content_hash,
                    imported_at, status, error, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')
                ON CONFLICT(id) DO UPDATE SET
                    filename = excluded.filename,
                    original_path = excluded.original_path,
                    file_type = excluded.file_type,
                    file_size = excluded.file_size,
                    status = excluded.status,
                    error = excluded.error
                """,
                (
                    source_id,
                    path.name,
                    str(path),
                    file_type,
                    file_size,
                    file_hash,
                    existing.imported_at if existing else self._now(),
                    status,
                    error,
                ),
            )

    def list_sources(self, limit: int = 20, offset: int = 0) -> list[SourceRecord]:
        rows = self._connection.execute(
            "SELECT * FROM documents ORDER BY imported_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [self._source(row) for row in rows]

    def source_count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM documents").fetchone()
        return int(row[0])

    def indexed_source_count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM documents WHERE status = 'indexed'"
        ).fetchone()
        return int(row[0])

    def chunk_count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.status = 'indexed'"
        ).fetchone()
        return int(row[0])

    def current_hashes(self, source_ids: Sequence[str]) -> dict[str, str]:
        if not source_ids:
            return {}
        placeholders = ",".join("?" for _ in source_ids)
        rows = self._connection.execute(
            f"SELECT id, content_hash FROM documents WHERE status = 'indexed' "
            f"AND id IN ({placeholders})",
            tuple(source_ids),
        ).fetchall()
        return {row["id"]: row["content_hash"] for row in rows}

    def current_sources(self, source_ids: Sequence[str]) -> dict[str, SourceRecord]:
        if not source_ids:
            return {}
        placeholders = ",".join("?" for _ in source_ids)
        rows = self._connection.execute(
            f"SELECT * FROM documents WHERE status = 'indexed' AND id IN ({placeholders})",
            tuple(source_ids),
        ).fetchall()
        return {row["id"]: self._source(row) for row in rows}

    def mark_removing(self, source_id: str) -> None:
        with self._connection:
            self._connection.execute(
                "UPDATE documents SET status = 'removing' WHERE id = ?", (source_id,)
            )

    def finish_removing(self, source_id: str) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM documents WHERE id = ?", (source_id,))

    def sources_to_recover(self) -> list[str]:
        rows = self._connection.execute(
            "SELECT id FROM documents WHERE status = 'removing'"
        ).fetchall()
        return [row["id"] for row in rows]

    def clear_sources(self) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM documents")
            self._connection.execute("DELETE FROM rag_metadata")

    def mark_reset_pending(self) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO rag_metadata(key, value) VALUES ('reset_pending', 'true') "
                "ON CONFLICT(key) DO UPDATE SET value = 'true'"
            )

    def reset_pending(self) -> bool:
        row = self._connection.execute(
            "SELECT value FROM rag_metadata WHERE key = 'reset_pending'"
        ).fetchone()
        return row is not None and row["value"] == "true"

    def all_source_ids(self) -> list[str]:
        rows = self._connection.execute("SELECT id FROM documents").fetchall()
        return [row["id"] for row in rows]

    def set_collection_metadata(self, model: str, dimensions: int) -> None:
        with self._connection:
            self._connection.executemany(
                "INSERT INTO rag_metadata(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (("embedding_model", model), ("vector_dimensions", str(dimensions))),
            )

    def collection_metadata(self) -> dict[str, str]:
        rows = self._connection.execute(
            "SELECT key, value FROM rag_metadata"
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}
