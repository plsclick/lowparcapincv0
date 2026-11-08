from pathlib import Path
import tempfile
import unittest
import uuid

from qdrant_client import models

from lowparllm.ai.rag.vector_store import VectorStore, VectorStoreError


class VectorStoreTests(unittest.TestCase):
    def test_collection_upsert_search_and_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "qdrant"
            store = VectorStore(path, "knowledge")
            point_id = str(uuid.uuid4())
            store.upsert(
                [
                    models.PointStruct(
                        id=point_id,
                        vector=[1.0, 0.0, 0.0],
                        payload={
                            "document_id": "doc-1",
                            "file_hash": "file-hash",
                            "content_hash": "chunk-hash",
                            "embedding_model": "local-embed",
                            "vector_dimensions": 3,
                            "text": "network protocol",
                            "filename": "notes.txt",
                        },
                    )
                ],
                model="local-embed",
                dimensions=3,
            )
            hits = store.search([1.0, 0.0, 0.0], limit=3, score_threshold=0.3)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].payload["filename"], "notes.txt")
            self.assertGreaterEqual(hits[0].score, 0.3)
            store.close()

            reopened = VectorStore(path, "knowledge")
            self.assertEqual(reopened.collection_info()["points"], 1)
            reopened.ensure_collection("local-embed", 3)
            with self.assertRaises(VectorStoreError):
                reopened.ensure_collection("different-model", 3)
            with self.assertRaises(VectorStoreError):
                reopened.ensure_collection("local-embed", 4)
            reopened.close()

    def test_delete_document_and_reset_only_touch_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = VectorStore(Path(temporary) / "qdrant", "knowledge")
            store.upsert(
                [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=[1.0, 0.0],
                        payload={
                            "document_id": "selected",
                            "file_hash": "a",
                            "content_hash": "b",
                            "embedding_model": "local",
                            "vector_dimensions": 2,
                            "text": "text",
                        },
                    )
                ],
                model="local",
                dimensions=2,
            )
            store.delete_document("selected")
            self.assertEqual(store.collection_info()["points"], 0)
            store.reset()
            self.assertFalse(store.collection_info()["exists"])
            store.close()

    def test_vector_dimension_is_validated_before_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = VectorStore(Path(temporary) / "qdrant", "knowledge")
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=[1.0, 2.0],
                payload={"embedding_model": "local", "vector_dimensions": 3},
            )
            with self.assertRaises(VectorStoreError):
                store.upsert([point], model="local", dimensions=3)
            store.close()


if __name__ == "__main__":
    unittest.main()
