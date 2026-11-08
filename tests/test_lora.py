import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lowparllm.ai.lora.adapters import AdapterManager
from lowparllm.ai.lora.datasets import (
    DatasetValidationError,
    import_dataset,
    list_datasets,
    validate_dataset,
)
from lowparllm.ai.lora.training import (
    TrainingConfig,
    TrainingError,
    TrainingManager,
    resource_check,
)
from lowparllm.ai.lora.worker import _encode_records


class DatasetTests(unittest.TestCase):
    def test_validate_instruction_jsonl_and_reject_unhashable_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "examples.jsonl"
            path.write_text(
                json.dumps({"instruction": "Question", "output": "Answer"})
                + "\n"
                + json.dumps(
                    {
                        "messages": [
                            {"role": [], "content": "bad"},
                            {"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "hello"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = validate_dataset(path)
            self.assertEqual(report.records, 2)
            self.assertEqual(report.valid_records, 1)
            self.assertIn("message role", report.errors[0])

    def test_explicit_import_and_listing_use_user_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            source = home / "examples.json"
            source.write_text(
                json.dumps(
                    [{"messages": [
                        {"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "Answer"},
                    ]}]
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"HOME": temporary, "XDG_DATA_HOME": str(home / "data")},
                clear=True,
            ):
                imported = import_dataset(source, name="qa")
                self.assertEqual(imported.records, 1)
                self.assertEqual(list_datasets(), (imported,))

    def test_import_rejects_invalid_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bad.json"
            source.write_text('{"instruction": "", "output": "answer"}', encoding="utf-8")
            with self.assertRaises(DatasetValidationError):
                import_dataset(source)


class TrainingConfigTests(unittest.TestCase):
    def test_config_requires_local_hugging_face_weights_and_imported_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "checkpoint"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            data = root / "data.jsonl"
            data.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hi"},
                            {"role": "assistant", "content": "Hello"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"HOME": temporary, "XDG_DATA_HOME": str(root / "xdg")},
                clear=True,
            ):
                dataset = import_dataset(data, name="chat")
                config = TrainingConfig(str(model), dataset.name).validate()
                self.assertEqual(config.method, "lora")
                with self.assertRaisesRegex(TrainingError, "Ollama model tag"):
                    TrainingConfig("qwen3.5:4b", dataset.name).validate()
                with self.assertRaisesRegex(TrainingError, "Batch size"):
                    TrainingConfig(str(model), dataset.name, batch_size=0).validate()

    def test_stale_worker_is_reported_as_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            with patch.dict(
                os.environ,
                {"HOME": temporary, "XDG_DATA_HOME": str(home / "data")},
                clear=True,
            ):
                manager = TrainingManager()
                manager._prepare_directories()
                job_dir = manager.jobs_dir / "012345abcdef"
                job_dir.mkdir()
                (job_dir / "status.json").write_text(
                    json.dumps(
                        {
                            "job_id": job_dir.name,
                            "state": "running",
                            "pid": 2147483647,
                            "config": {},
                        }
                    ),
                    encoding="utf-8",
                )
                status = manager.status(job_dir.name)[0]
                self.assertEqual(status["state"], "interrupted")

    def test_start_reports_missing_optional_backend_without_fake_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "checkpoint"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            source = root / "data.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hi"},
                            {"role": "assistant", "content": "Hello"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"HOME": temporary, "XDG_DATA_HOME": str(root / "xdg")},
                clear=True,
            ):
                dataset = import_dataset(source, name="chat")
                manager = TrainingManager()
                with patch.object(
                    manager, "missing_dependencies", return_value=("torch",)
                ):
                    with self.assertRaisesRegex(TrainingError, "Optional training packages"):
                        manager.start(TrainingConfig(str(model), dataset.name))
                self.assertEqual(manager.status(), ())

    def test_qlora_preflight_rejects_missing_nvidia_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "checkpoint"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            data = root / "data.jsonl"
            data.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hi"},
                            {"role": "assistant", "content": "Hello"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"HOME": temporary, "XDG_DATA_HOME": str(root / "xdg")},
                clear=True,
            ):
                dataset = import_dataset(data, name="chat")
                config = TrainingConfig(str(model), dataset.name, method="qlora")
                with patch("lowparllm.ai.lora.training.shutil.which", return_value=None):
                    with self.assertRaisesRegex(TrainingError, "NVIDIA CUDA"):
                        resource_check(config)


class AdapterTests(unittest.TestCase):
    def test_delete_requires_confirmation_and_preserves_other_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            with patch.dict(
                os.environ,
                {"HOME": temporary, "XDG_DATA_HOME": str(home / "data")},
                clear=True,
            ):
                manager = AdapterManager()
                adapter_dir = manager.directory / "012345abcdef"
                adapter_dir.mkdir(parents=True)
                (adapter_dir / "adapter.safetensors").write_bytes(b"adapter")
                (adapter_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "name": adapter_dir.name,
                            "base_model_path": "/local/checkpoint",
                            "status": "completed",
                            "adapter_path": str(adapter_dir),
                        }
                    ),
                    encoding="utf-8",
                )
                unrelated = manager.directory / "keep.txt"
                unrelated.write_text("keep", encoding="utf-8")
                with self.assertRaisesRegex(TrainingError, "explicit confirmation"):
                    manager.delete(adapter_dir.name)
                manager.delete(adapter_dir.name, confirmed=True)
                self.assertTrue(unrelated.exists())
                self.assertFalse(adapter_dir.exists())

    def test_activation_does_not_claim_ollama_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(
                os.environ,
                {"HOME": temporary, "XDG_DATA_HOME": str(Path(temporary) / "data")},
                clear=True,
            ):
                manager = AdapterManager()
                adapter_dir = manager.directory / "012345abcdef"
                adapter_dir.mkdir(parents=True)
                (adapter_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "name": adapter_dir.name,
                            "base_model_path": "/local/checkpoint",
                            "status": "completed",
                            "adapter_path": str(adapter_dir),
                        }
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(TrainingError, "cannot safely activate"):
                    manager.activate(adapter_dir.name, "qwen3.5:4b")


class _FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        del tokenize
        token_ids = [
            len(message["content"]) + index
            for index, message in enumerate(messages)
        ]
        if add_generation_prompt and messages and messages[-1]["role"] == "user":
            token_ids.append(100)
        elif not add_generation_prompt and messages and messages[-1]["role"] == "assistant":
            token_ids.insert(-1, 100)
        return token_ids


class WorkerEncodingTests(unittest.TestCase):
    def test_assistant_only_labels_and_sequence_truncation(self) -> None:
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        ]
        encoded = _encode_records(_FakeTokenizer(), examples, max_length=2)
        self.assertEqual(len(encoded), 1)
        self.assertEqual(len(encoded[0]["input_ids"]), 2)
        self.assertEqual(len(encoded[0]["labels"]), 2)
        self.assertTrue(any(label != -100 for label in encoded[0]["labels"]))


if __name__ == "__main__":
    unittest.main()
