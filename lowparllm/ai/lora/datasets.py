"""Explicit import and validation for conversational fine-tuning datasets."""

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from lowparllm.core.paths import data_dir

_MAX_DATASET_BYTES = 256 * 1024 * 1024
_DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class DatasetValidationError(ValueError):
    """A selected training dataset is malformed or unsupported."""


@dataclass(frozen=True)
class DatasetReport:
    path: Path
    format: str
    records: int
    valid_records: int
    duplicates: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ImportedDataset:
    name: str
    path: Path
    records: int
    format: str
    digest: str


def datasets_dir() -> Path:
    training = data_dir() / "training"
    if training.is_symlink():
        raise DatasetValidationError("Refusing to use a symbolic-link training directory.")
    return training / "datasets"


def validate_dataset(path: Path) -> DatasetReport:
    target = path.expanduser()
    if target.is_symlink() or not target.is_file():
        raise DatasetValidationError("Dataset must be a regular, non-symlink file.")
    if target.stat().st_size > _MAX_DATASET_BYTES:
        raise DatasetValidationError("Dataset exceeds the 256 MiB validation limit.")
    suffix = target.suffix.lower()
    try:
        if suffix in {".jsonl", ".ndjson"}:
            records, errors = _read_jsonl(target)
            format_name = "jsonl"
        elif suffix == ".json":
            records, errors = _read_json(target)
            format_name = "json"
        elif suffix == ".csv":
            records, errors = _read_csv(target)
            format_name = "csv"
        else:
            raise DatasetValidationError("Supported dataset formats: JSONL, JSON, CSV.")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError("Dataset must be UTF-8 encoded.") from exc
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    for index, record in enumerate(records, start=1):
        messages, error = _normalize_record(record)
        if error is not None:
            errors.append(f"Record {index}: {error}")
            continue
        signature = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        valid.append({"messages": messages})
    return DatasetReport(
        path=target.resolve(),
        format=format_name,
        records=len(records),
        valid_records=len(valid),
        duplicates=duplicates,
        errors=tuple(errors),
    )


def load_validated_records(path: Path) -> list[dict[str, Any]]:
    target = path.expanduser()
    suffix = target.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        records, errors = _read_jsonl(target)
    elif suffix == ".json":
        records, errors = _read_json(target)
    elif suffix == ".csv":
        records, errors = _read_csv(target)
    else:
        raise DatasetValidationError("Unsupported training dataset format.")
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        messages, error = _normalize_record(record)
        if error:
            raise DatasetValidationError(f"Record {index}: {error}")
        signature = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            valid.append({"messages": messages})
    if errors:
        raise DatasetValidationError(errors[0])
    if not valid:
        raise DatasetValidationError("Dataset contains no valid unique records.")
    return valid


def import_dataset(path: Path, *, name: str | None = None) -> ImportedDataset:
    source = path.expanduser()
    report = validate_dataset(source)
    if report.errors:
        raise DatasetValidationError(
            "Dataset has invalid records; fix them before importing: "
            + report.errors[0]
        )
    if report.valid_records == 0:
        raise DatasetValidationError("Dataset contains no valid examples.")
    normalized_name = name or _slug(source.stem)
    if not _DATASET_NAME.fullmatch(normalized_name):
        raise DatasetValidationError(
            "Dataset name must use 1–64 letters, numbers, dots, underscores or hyphens."
        )
    directory = datasets_dir()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if directory.is_symlink() or directory.parent.is_symlink():
        raise DatasetValidationError("Refusing to use a symbolic-link dataset directory.")
    raw_digest = _file_digest(source)
    target = directory / f"{normalized_name}{source.suffix.lower()}"
    if target.exists():
        target = directory / f"{normalized_name}-{raw_digest[:8]}{source.suffix.lower()}"
    if target.exists():
        raise DatasetValidationError(f"Dataset {target.stem!r} is already imported.")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".dataset-", dir=directory
    )
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary_name)
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, target)
    except OSError:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    metadata = {
        "name": target.stem,
        "source_name": source.name,
        "format": report.format,
        "records": report.valid_records,
        "sha256": raw_digest,
    }
    metadata_path = target.with_suffix(target.suffix + ".lowp.json")
    try:
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.chmod(metadata_path, 0o600)
    except OSError:
        target.unlink(missing_ok=True)
        raise
    return ImportedDataset(
        name=target.stem,
        path=target,
        records=report.valid_records,
        format=report.format,
        digest=raw_digest,
    )


def list_datasets() -> tuple[ImportedDataset, ...]:
    directory = datasets_dir()
    if not directory.exists():
        return ()
    if directory.is_symlink() or directory.parent.is_symlink():
        raise DatasetValidationError("Refusing to use a symbolic-link dataset directory.")
    items: list[ImportedDataset] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name.endswith(".lowp.json") or path.suffix.lower() not in {
            ".jsonl", ".ndjson", ".json", ".csv"
        }:
            continue
        report = validate_dataset(path)
        items.append(
            ImportedDataset(
                name=path.stem,
                path=path,
                records=report.valid_records,
                format=report.format,
                digest=_file_digest(path),
            )
        )
    return tuple(items)


def get_dataset(name: str) -> ImportedDataset:
    if not _DATASET_NAME.fullmatch(name):
        raise DatasetValidationError("Invalid imported dataset name.")
    matches = [dataset for dataset in list_datasets() if dataset.name == name]
    if len(matches) != 1:
        raise DatasetValidationError(f"Imported dataset {name!r} was not found.")
    dataset = matches[0]
    report = validate_dataset(dataset.path)
    if report.errors:
        raise DatasetValidationError(
            f"Imported dataset {name!r} is no longer valid: {report.errors[0]}"
        )
    if report.valid_records == 0:
        raise DatasetValidationError(f"Imported dataset {name!r} has no valid records.")
    return dataset


def _read_jsonl(path: Path) -> tuple[list[object], list[str]]:
    records: list[object] = []
    errors: list[str] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_number}: invalid JSON ({exc.msg}).")
    return records, errors


def _read_json(path: Path) -> tuple[list[object], list[str]]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"Invalid JSON: {exc.msg}.") from exc
    if isinstance(payload, list):
        return payload, []
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"], []
    if isinstance(payload, dict):
        return [payload], []
    raise DatasetValidationError("JSON dataset must contain an object or array of objects.")


def _read_csv(path: Path) -> tuple[list[object], list[str]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise DatasetValidationError("CSV dataset must have a header row.")
        return list(reader), []


def _normalize_record(record: object) -> tuple[list[dict[str, str]], str | None]:
    if not isinstance(record, dict):
        return [], "record must be an object."
    raw_messages = record.get("messages", record.get("conversations"))
    if isinstance(raw_messages, list):
        messages: list[dict[str, str]] = []
        for message in raw_messages:
            if not isinstance(message, dict):
                return [], "messages must contain objects."
            role = message.get("role", message.get("from"))
            content = message.get("content", message.get("value"))
            if not isinstance(role, str):
                return [], "message role must be system, user, or assistant."
            if role in {"human", "user"}:
                role = "user"
            elif role in {"gpt", "assistant"}:
                role = "assistant"
            elif role == "system":
                role = "system"
            else:
                return [], "message role must be system, user, or assistant."
            if not isinstance(content, str) or not content.strip():
                return [], "message content must be non-empty text."
            messages.append({"role": role, "content": content.strip()})
    else:
        instruction = record.get("instruction", record.get("prompt"))
        output = record.get("output", record.get("completion", record.get("response")))
        context = record.get("input", "")
        if not isinstance(instruction, str) or not instruction.strip():
            return [], "record needs messages or a non-empty instruction/prompt."
        if not isinstance(output, str) or not output.strip():
            return [], "record needs a non-empty output/completion/response."
        if not isinstance(context, str):
            return [], "input/context must be text."
        user_content = instruction.strip()
        if context.strip():
            user_content += "\n\n" + context.strip()
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output.strip()},
        ]
    if not any(item["role"] == "user" for item in messages):
        return [], "conversation needs at least one user message."
    if not any(item["role"] == "assistant" for item in messages):
        return [], "conversation needs at least one assistant message."
    return messages, None


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return (normalized or "dataset")[:64]
