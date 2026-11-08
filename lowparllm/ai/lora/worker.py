"""Isolated optional Transformers/PEFT training worker."""

import importlib
import logging
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from typing import Any

from lowparllm.ai.lora.datasets import get_dataset, load_validated_records
from lowparllm.ai.lora.training import (
    TrainingConfig,
    TrainingError,
    _read_json,
    _write_json,
)

logger = logging.getLogger("lowparllm.training")


class TrainingCancelled(Exception):
    """The user requested cancellation of the current training run."""


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m lowparllm.ai.lora.worker <job.json>", file=sys.stderr)
        return 2
    spec_path = Path(sys.argv[1])
    spec: dict[str, Any] = {}
    try:
        spec = _read_json(spec_path)
        return train_job(spec)
    except TrainingCancelled:
        logger.info("Training cancelled.")
        _set_job_state(spec, "cancelled")
        return 130
    except Exception:
        logger.exception("Training failed.")
        try:
            status_path = Path(spec.get("job_dir", "")) / "status.json"
            status = _read_json(status_path)
            status.update(
                {
                    "state": "failed",
                    "error": traceback.format_exc(limit=12)[-8000:],
                    "finished_at": time.time(),
                }
            )
            _write_json(status_path, status)
        except (TrainingError, OSError, TypeError):
            pass
        return 1


def train_job(spec: dict[str, Any]) -> int:
    job_dir = Path(spec.get("job_dir", ""))
    status_path = job_dir / "status.json"
    status = _read_json(status_path)
    _install_signal_handlers()
    configuration = TrainingConfig(
        base_model_path=str(spec.get("base_model_path", "")),
        dataset_name=str(spec.get("dataset_name", "")),
        method=str(spec.get("method", "lora")),
        epochs=float(spec.get("epochs", 1.0)),
        learning_rate=float(spec.get("learning_rate", 0.0002)),
        batch_size=int(spec.get("batch_size", 1)),
        gradient_accumulation=int(spec.get("gradient_accumulation", 8)),
        max_sequence_length=int(spec.get("max_sequence_length", 1024)),
        lora_rank=int(spec.get("lora_rank", 8)),
        lora_alpha=int(spec.get("lora_alpha", 16)),
        lora_dropout=float(spec.get("lora_dropout", 0.05)),
        checkpoint_steps=int(spec.get("checkpoint_steps", 50)),
        output_directory=spec.get("output_directory"),
    ).validate()
    dataset = get_dataset(configuration.dataset_name)
    records = load_validated_records(dataset.path)
    if not records:
        raise TrainingError("The selected dataset has no valid examples.")

    torch = importlib.import_module("torch")
    peft = importlib.import_module("peft")
    transformers = importlib.import_module("transformers")
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer
    Trainer = transformers.Trainer
    TrainingArguments = transformers.TrainingArguments

    base_path = str(Path(configuration.base_model_path).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise TrainingError(
            "The selected checkpoint has no chat template; LowP cannot safely "
            "format conversational examples for it."
        )
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if configuration.method == "qlora":
        model_kwargs.update(
            {
                "quantization_config": transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                ),
                "device_map": {"": 0},
            }
        )
    elif torch.cuda.is_available():
        model_kwargs.update(
            {"torch_dtype": torch.float16, "device_map": "auto"}
        )
    else:
        model_kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_path, **model_kwargs)
    model.config.use_cache = False
    if configuration.method == "qlora":
        model = prepare_model_for_kbit_training(model)
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=configuration.lora_rank,
            lora_alpha=configuration.lora_alpha,
            lora_dropout=configuration.lora_dropout,
            bias="none",
            task_type=peft.TaskType.CAUSAL_LM,
            target_modules="all-linear",
        ),
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    encoded = _encode_records(tokenizer, records, configuration.max_sequence_length)
    if not encoded:
        raise TrainingError("No assistant responses fit the selected sequence length.")
    train_dataset = _TorchDataset(encoded, torch)
    collator = _Collator(tokenizer, torch)
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    status.update({"state": "running", "pid": os.getpid(), "started_at": time.time()})
    _write_json(status_path, status)
    progress_callback = _make_progress_callback(status_path)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=configuration.epochs,
        learning_rate=configuration.learning_rate,
        per_device_train_batch_size=configuration.batch_size,
        gradient_accumulation_steps=configuration.gradient_accumulation,
        max_steps=-1,
        logging_steps=1,
        save_strategy="steps",
        save_steps=configuration.checkpoint_steps,
        save_total_limit=3,
        fp16=torch.cuda.is_available() and configuration.method == "lora",
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        optim="paged_adamw_8bit" if configuration.method == "qlora" else "adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[progress_callback],
    )
    resume = bool(spec.get("resume"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    adapter_dir = Path(spec["adapter_dir"])
    if adapter_dir.exists():
        raise TrainingError("Refusing to overwrite an existing adapter directory.")
    adapter_dir.mkdir(parents=True, mode=0o700)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    metadata = {
        "name": adapter_dir.name,
        "base_model_path": base_path,
        "training_method": configuration.method,
        "dataset_name": dataset.name,
        "dataset_sha256": dataset.digest,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_config": {
            key: value
            for key, value in spec.items()
            if key
            in {
                "method", "epochs", "learning_rate", "batch_size",
                "gradient_accumulation", "max_sequence_length", "lora_rank",
                "lora_alpha", "lora_dropout", "checkpoint_steps",
                "output_directory",
            }
        },
        "adapter_path": str(adapter_dir),
        "status": "completed",
        "ollama_compatible": False,
        "compatibility": (
            "PEFT adapter for a Hugging Face checkpoint; Ollama cannot activate "
            "this adapter directly."
        ),
    }
    _write_json(adapter_dir / "metadata.json", metadata)
    status.update(
        {
            "state": "completed",
            "finished_at": time.time(),
            "adapter_path": str(adapter_dir),
        }
    )
    _write_json(status_path, status)
    logger.info("Training completed. Adapter saved to %s", adapter_dir)
    return 0


def prepare_model_for_kbit_training(model: Any) -> Any:
    try:
        prepare = importlib.import_module("peft").prepare_model_for_kbit_training
    except ImportError as exc:
        raise TrainingError("Installed PEFT package lacks QLoRA preparation support.") from exc
    return prepare(model, use_gradient_checkpointing=True)


def _encode_records(
    tokenizer: Any, records: list[dict[str, Any]], max_length: int
) -> list[dict[str, list[int]]]:
    encoded: list[dict[str, list[int]]] = []
    for record in records:
        messages = record["messages"]
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            if not any(item["role"] == "user" for item in messages[:index]):
                continue
            prompt = tokenizer.apply_chat_template(
                messages[:index],
                tokenize=True,
                add_generation_prompt=True,
            )
            full = tokenizer.apply_chat_template(
                messages[: index + 1],
                tokenize=True,
                add_generation_prompt=False,
            )
            if not isinstance(prompt, list) or not isinstance(full, list):
                raise TrainingError("Tokenizer chat template returned invalid token IDs.")
            if any(not isinstance(token, int) for token in prompt + full):
                raise TrainingError("Tokenizer chat template returned non-integer token IDs.")
            if full[: len(prompt)] != prompt:
                raise TrainingError(
                    "Tokenizer chat template does not preserve the prompt prefix; "
                    "assistant-only loss cannot be applied safely."
                )
            if len(full) > max_length:
                trim = len(full) - max_length
                if trim >= len(prompt):
                    continue
                prompt = prompt[trim:]
                full = full[-max_length:]
            if len(full) <= len(prompt):
                continue
            prefix_length = min(len(prompt), len(full))
            labels = [-100] * prefix_length + full[prefix_length:]
            if any(label != -100 for label in labels):
                encoded.append({"input_ids": full, "labels": labels})
    return encoded


class _TorchDataset:
    def __init__(self, examples: list[dict[str, list[int]]], torch: Any) -> None:
        self.examples = examples
        self.torch = torch

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        return {
            "input_ids": self.torch.tensor(example["input_ids"], dtype=self.torch.long),
            "labels": self.torch.tensor(example["labels"], dtype=self.torch.long),
        }


class _Collator:
    def __init__(self, tokenizer: Any, torch: Any) -> None:
        self.tokenizer = tokenizer
        self.torch = torch
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise TrainingError("Tokenizer has no pad or EOS token for batching.")
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        labels = []
        attention = []
        for item in features:
            padding = max_length - len(item["input_ids"])
            input_ids.append(
                self.torch.nn.functional.pad(item["input_ids"], (0, padding), value=pad_id)
            )
            labels.append(
                self.torch.nn.functional.pad(item["labels"], (0, padding), value=-100)
            )
            attention.append(
                self.torch.nn.functional.pad(
                    self.torch.ones_like(item["input_ids"]), (0, padding), value=0
                )
            )
        return {
            "input_ids": self.torch.stack(input_ids),
            "labels": self.torch.stack(labels),
            "attention_mask": self.torch.stack(attention),
        }


def _make_progress_callback(status_path: Path) -> Any:
    TrainerCallback = importlib.import_module("transformers").TrainerCallback

    class ProgressCallback(TrainerCallback):
        def on_log(self, _args, state, control, logs=None, **_kwargs):
            del _args, _kwargs
            status = _read_json(status_path)
            status.update(
                {
                    "state": "running",
                    "step": state.global_step,
                    "max_steps": state.max_steps,
                    "metrics": logs or {},
                    "updated_at": time.time(),
                }
            )
            _write_json(status_path, status)
            logger.info(
                "Training step %s/%s: %s", state.global_step, state.max_steps, logs
            )
            return control

        def on_save(self, args, state, control, **_kwargs):
            del _kwargs
            status = _read_json(status_path)
            status.update(
                {
                    "state": "running",
                    "checkpoint": str(
                        Path(args.output_dir) / f"checkpoint-{state.global_step}"
                    ),
                    "updated_at": time.time(),
                }
            )
            _write_json(status_path, status)
            return control

    return ProgressCallback()


def _set_job_state(spec: dict[str, Any], state_name: str) -> None:
    try:
        status_path = Path(spec["job_dir"]) / "status.json"
        status = _read_json(status_path)
        status.update({"state": state_name, "finished_at": time.time()})
        _write_json(status_path, status)
    except (KeyError, TrainingError, OSError):
        logger.exception("Could not save final training status.")


def _install_signal_handlers() -> None:
    def cancel(_signum, _frame):
        del _signum, _frame
        raise TrainingCancelled()

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, cancel)


if __name__ == "__main__":
    raise SystemExit(main())
