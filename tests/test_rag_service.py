import asyncio
from collections.abc import Sequence
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lowparllm.ai.embeddings.ollama_embeddings import EmbeddingModelMissing
from lowparllm.ai.rag.rag_service import RagService
from lowparllm.core.config import AppConfig, RagConfig, _default_config_text


class FakeEmbeddings:
    def __init__(self, *, available: bool = True, batch_size: int = 2):
        self.available = available
        self.batch_size = batch_size
        self.calls = 0

    async def is_available(self, *, refresh: bool = False) -> bool:
        del refresh
        return self.available

    async def aclose(self) -> None:
        return None

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [1.0, 0.0, 0.0] if "network" in text.lower() else [0.0, 1.0, 0.0]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_batches(self, texts: Sequence[str]):
        if not self.available:
            raise EmbeddingModelMissing(
                "Embedding model 'fake-embed' is not installed. "
                "Install it manually with: ollama pull fake-embed."
            )
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset : offset + self.batch_size]
            self.calls += 1
            yield [self._vector(text) for text in batch]


class RagServiceTests(unittest.TestCase):
    def test_automatic_retrieval_is_off_by_default(self) -> None:
        async def run() -> None:
            config = AppConfig()
            self.assertFalse(config.rag.automatic_retrieval)
            self.assertIn("automatic_retrieval = false", _default_config_text())
            service = RagService(config, embeddings=FakeEmbeddings())

            self.assertFalse(await service.has_indexed_sources())
            context = await service.context_for_chat("hey")
            self.assertEqual(context.text, "")
            self.assertEqual(context.citations, ())
            await service.close()

        asyncio.run(run())

    def test_index_deduplicate_reindex_retrieve_and_restart(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / "networking notes.md"
                path.write_text(
                    "# TCP\n\nNetwork congestion control uses slow start and "
                    "congestion avoidance.",
                    encoding="utf-8",
                )
                config = AppConfig(
                    rag=RagConfig(
                        automatic_retrieval=True,
                        embedding_model="fake-embed",
                        chunk_size=100,
                        chunk_overlap=10,
                        qdrant_path=root / "qdrant",
                        sqlite_path=root / "knowledge.db",
                    )
                )
                embeddings = FakeEmbeddings()
                service = RagService(config, embeddings=embeddings)
                first = await service.add_file(path)
                self.assertEqual(first.status, "indexed")
                self.assertEqual(embeddings.calls, 1)

                duplicate = await service.add_file(path)
                self.assertEqual(duplicate.status, "skipped")
                self.assertEqual(embeddings.calls, 1)

                context = await service.context_for_chat("Explain network congestion")
                self.assertIn("congestion avoidance", context.text)
                self.assertEqual(context.citations[0].filename, path.name)
                results = await service.search("network")
                self.assertEqual(len(results.chunks), 1)
                self.assertEqual(results.chunks[0].payload["filename"], path.name)

                status = await service.status()
                self.assertEqual(status["indexed_documents"], 1)
                self.assertEqual(status["chunks"], 1)
                self.assertEqual(status["vector_dimensions"], "3")
                await service.close()

                reopened = RagService(config, embeddings=FakeEmbeddings())
                reopened_status = await reopened.status()
                self.assertEqual(reopened_status["indexed_documents"], 1)
                self.assertEqual(reopened_status["vectors"], 1)
                reindexed = await reopened.reindex(first.source_id)
                self.assertEqual(reindexed.status, "indexed")
                self.assertEqual((await reopened.status())["vectors"], 1)
                removed = await reopened.remove_source(first.source_id)
                self.assertEqual(removed.filename, path.name)
                self.assertTrue(path.exists())
                self.assertEqual((await reopened.status())["indexed_documents"], 0)
                await reopened.close()

        asyncio.run(run())

    def test_missing_embedding_model_does_not_remove_existing_chat_support(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / "notes.txt"
                path.write_text("A local networking note.", encoding="utf-8")
                config = AppConfig(
                    rag=RagConfig(
                        embedding_model="fake-embed",
                        qdrant_path=root / "qdrant",
                        sqlite_path=root / "knowledge.db",
                    )
                )
                service = RagService(
                    config, embeddings=FakeEmbeddings(available=False)
                )
                result = await service.add_file(path)
                self.assertEqual(result.status, "failed")
                self.assertIn("ollama pull fake-embed", result.message)
                await service.close()

        asyncio.run(run())

    def test_duplicate_content_moves_source_without_regenerating_vectors(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                first_path, moved_path = root / "one.md", root / "renamed.md"
                first_path.write_text("network notes", encoding="utf-8")
                config = AppConfig(
                    rag=RagConfig(
                        embedding_model="fake-embed",
                        qdrant_path=root / "qdrant",
                        sqlite_path=root / "knowledge.db",
                    )
                )
                embeddings = FakeEmbeddings()
                service = RagService(config, embeddings=embeddings)
                original = await service.add_file(first_path)
                first_path.rename(moved_path)
                moved = await service.add_file(moved_path)
                self.assertEqual(moved.status, "skipped")
                self.assertEqual(moved.source_id, original.source_id)
                self.assertEqual(embeddings.calls, 1)
                result = await service.search("network")
                self.assertEqual(result.chunks[0].payload["filename"], moved_path.name)
                await service.close()

        asyncio.run(run())

    def test_modified_file_is_incrementally_reindexed(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / "notes.md"
                path.write_text("Network plans use slow start.", encoding="utf-8")
                config = AppConfig(
                    rag=RagConfig(
                        embedding_model="fake-embed",
                        qdrant_path=root / "qdrant",
                        sqlite_path=root / "knowledge.db",
                    )
                )
                service = RagService(config, embeddings=FakeEmbeddings())
                original = await service.add_file(path)
                path.write_text("Network plans use congestion avoidance.", encoding="utf-8")
                updated = await service.add_file(path)

                self.assertEqual(updated.status, "indexed")
                self.assertEqual(updated.source_id, original.source_id)
                status = await service.status()
                self.assertEqual(status["indexed_documents"], 1)
                self.assertEqual(status["vectors"], 1)
                result = await service.search("network")
                self.assertIn("congestion avoidance", result.chunks[0].payload["text"])
                await service.close()

        asyncio.run(run())

    def test_folder_import_reports_per_file_failures_and_progress(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "valid.txt").write_text("Local notes.", encoding="utf-8")
                (root / "empty.pdf").write_bytes(b"")
                config = AppConfig(
                    rag=RagConfig(
                        embedding_model="fake-embed",
                        qdrant_path=root / "qdrant",
                        sqlite_path=root / "knowledge.db",
                    )
                )
                service = RagService(config, embeddings=FakeEmbeddings())
                progress = []
                files, results = await service.add_folder(
                    root, lambda current, total, path: progress.append(
                        (current, total, path.name)
                    )
                )

                self.assertEqual(len(files), 2)
                self.assertEqual(
                    {result.status for _, result in results}, {"indexed", "failed"}
                )
                self.assertEqual(
                    progress,
                    [(1, 2, "empty.pdf"), (2, 2, "valid.txt")],
                )
                await service.close()

        asyncio.run(run())

    def test_identical_chunks_from_distinct_sources_keep_both_citations(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                first_path, second_path = root / "first.md", root / "second.md"
                first_path.write_text(
                    "Network context is useful.\n\nnetwork first.", encoding="utf-8"
                )
                second_path.write_text(
                    "Network context is useful.\n\nnetwork second.", encoding="utf-8"
                )
                config = AppConfig(
                    rag=RagConfig(
                        embedding_model="fake-embed",
                        chunk_size=4,
                        chunk_overlap=0,
                        qdrant_path=root / "qdrant",
                        sqlite_path=root / "knowledge.db",
                    )
                )
                service = RagService(config, embeddings=FakeEmbeddings())
                self.assertEqual((await service.add_file(first_path)).status, "indexed")
                self.assertEqual((await service.add_file(second_path)).status, "indexed")

                results = await service.search("network context")
                names = {chunk.payload["filename"] for chunk in results.chunks}
                self.assertEqual(names, {first_path.name, second_path.name})
                await service.close()

        asyncio.run(run())

    def test_unchanged_source_stays_indexed_when_embedding_model_is_unavailable(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / "notes.txt"
                path.write_text("A local networking note.", encoding="utf-8")
                config = AppConfig(
                    rag=RagConfig(
                        embedding_model="fake-embed",
                        qdrant_path=root / "qdrant",
                        sqlite_path=root / "knowledge.db",
                    )
                )
                embeddings = FakeEmbeddings()
                service = RagService(config, embeddings=embeddings)
                indexed = await service.add_file(path)
                embeddings.available = False

                repeated = await service.add_file(path)
                self.assertEqual(repeated.status, "skipped")
                self.assertEqual(repeated.source_id, indexed.source_id)
                current = await service.get_source(indexed.source_id)
                self.assertIsNotNone(current)
                assert current is not None
                self.assertEqual(current.status, "indexed")
                self.assertEqual(embeddings.calls, 1)
                await service.close()

        asyncio.run(run())

    def test_importing_a_symlink_does_not_mark_its_target_stale(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / "notes.txt"
                link = root / "linked-notes.txt"
                path.write_text("A local networking note.", encoding="utf-8")
                link.symlink_to(path)
                config = AppConfig(
                    rag=RagConfig(
                        embedding_model="fake-embed",
                        qdrant_path=root / "qdrant",
                        sqlite_path=root / "knowledge.db",
                    )
                )
                service = RagService(config, embeddings=FakeEmbeddings())
                indexed = await service.add_file(path)
                self.assertEqual(indexed.status, "indexed")

                linked = await service.add_file(link)
                current = await service.get_source(indexed.source_id)
                self.assertEqual(linked.status, "failed")
                self.assertIsNotNone(current)
                assert current is not None
                self.assertEqual(current.status, "indexed")
                await service.close()

        asyncio.run(run())

    def test_reset_and_pending_reset_recovery_preserve_original_documents(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / "notes.txt"
                path.write_text("A local networking note.", encoding="utf-8")
                config = AppConfig(
                    rag=RagConfig(
                        embedding_model="fake-embed",
                        qdrant_path=root / "qdrant",
                        sqlite_path=root / "knowledge.db",
                    )
                )
                service = RagService(config, embeddings=FakeEmbeddings())
                indexed = await service.add_file(path)
                self.assertEqual(indexed.status, "indexed")
                await service.reset_knowledge()
                self.assertTrue(path.exists())
                self.assertEqual((await service.status())["indexed_documents"], 0)

                next_indexed = await service.add_file(path)
                self.assertEqual(next_indexed.status, "indexed")
                assert service._registry is not None
                service._registry.mark_reset_pending()
                await service.close()

                reopened = RagService(config, embeddings=FakeEmbeddings())
                status = await reopened.status()
                self.assertEqual(status["indexed_documents"], 0)
                self.assertEqual(status["vectors"], 0)
                self.assertTrue(path.exists())
                await reopened.close()

        asyncio.run(run())

    def test_rag_toggle_persists_only_its_setting(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as temporary:
                config_path = Path(temporary) / "config"
                config_path.mkdir()
                config_file = config_path / "config.toml"
                original = _default_config_text().replace(
                    "automatic_retrieval = false",
                    "automatic_retrieval = true # keep this setting comment",
                ) + "\n# preserve me\n"
                config_file.write_text(original, encoding="utf-8")
                with patch("lowparllm.core.config.config_dir", return_value=config_path):
                    service = RagService(
                        AppConfig(rag=RagConfig(automatic_retrieval=True)),
                        embeddings=FakeEmbeddings(),
                    )
                    await service.set_automatic_retrieval(False)
                    updated = config_file.read_text(encoding="utf-8")
                    self.assertFalse(service.automatic_retrieval)
                    self.assertIn(
                        "automatic_retrieval = false # keep this setting comment",
                        updated,
                    )
                    self.assertIn("# preserve me", updated)
                    await service.close()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
