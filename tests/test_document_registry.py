from pathlib import Path
import tempfile
import unittest

from lowparllm.storage.database import ChunkRecord, DocumentRegistry


class DocumentRegistryTests(unittest.TestCase):
    def test_registry_persists_metadata_and_detects_duplicate_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "knowledge.db"
            registry = DocumentRegistry(database)
            chunks = [
                ChunkRecord("chunk-1", 0, "chunk-hash", 2, "Section", 10, 14)
            ]
            registry.register_indexed(
                source_id="source-1",
                path=Path(temporary) / "notes.md",
                file_type=".md",
                file_size=20,
                file_hash="file-hash",
                chunks=chunks,
                embedding_model="local-embed",
                metadata={"vector_dimensions": 3, "title": "Notes"},
            )
            self.assertEqual(registry.find_hash("file-hash").id, "source-1")
            self.assertEqual(registry.chunk_count(), 1)
            self.assertEqual(
                registry.collection_metadata(),
                {"embedding_model": "local-embed", "vector_dimensions": "3"},
            )
            registry.close()

            reopened = DocumentRegistry(database)
            source = reopened.get_source("source")
            self.assertEqual(source.filename, "notes.md")
            self.assertEqual(source.metadata["title"], "Notes")
            self.assertEqual(source.status, "indexed")
            reopened.close()

    def test_changed_index_failure_marks_source_stale_and_recovery_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = DocumentRegistry(Path(temporary) / "knowledge.db")
            path = Path(temporary) / "guide.txt"
            registry.register_indexed(
                source_id="source-1",
                path=path,
                file_type=".txt",
                file_size=8,
                file_hash="old",
                chunks=[
                    ChunkRecord("chunk-1", 0, "chunk-hash", None, None, 1, 1)
                ],
                embedding_model="local",
                metadata={"vector_dimensions": 2},
            )
            registry.record_failure(
                source_id="source-1",
                path=path,
                file_type=".txt",
                file_size=10,
                file_hash="new",
                error="embedding failed",
            )
            self.assertEqual(registry.get_source("source-1").status, "stale")
            self.assertEqual(registry.indexed_source_count(), 0)
            registry.mark_removing("source-1")
            self.assertEqual(registry.sources_to_recover(), ["source-1"])
            registry.finish_removing("source-1")
            self.assertEqual(registry.source_count(), 0)
            registry.close()

    def test_unrelated_database_file_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "knowledge.db"
            database.write_text("unrelated file", encoding="utf-8")
            original = database.read_bytes()
            with self.assertRaises(Exception):
                DocumentRegistry(database)
            self.assertEqual(database.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
