# Local training and adapters

## What this phase supports

LowP can launch optional local LoRA training through Transformers and PEFT.
QLoRA is limited to a detected NVIDIA CUDA workflow using bitsandbytes 4-bit
loading. Training runs in a separate worker process so the terminal remains
available. Jobs write status, logs, and Transformers checkpoints under
the per-user LowP data directory (`~/.local/share/lowp/training/` on Linux,
`%LOCALAPPDATA%\lowp\training\` on Windows).
The Windows worker lifecycle is implemented but has not been validated on a
native Windows host; see [Windows support](windows-support.md).

The worker loads only a local Hugging Face checkpoint directory containing a
`config.json` and recognized model weights. It sets
`local_files_only=True` and does not execute remote model code. An Ollama tag
such as `qwen3.5:4b` is not a training checkpoint and cannot be passed as
`--base`. Obtain and configure a compatible checkpoint separately; LowP does
not download model weights.

LoRA trains a small adapter while freezing the base weights. QLoRA additionally
quantizes the supported base checkpoint to 4-bit to reduce GPU memory use. RAG
is different: it retrieves explicitly indexed files at inference time and
does not train or modify model weights. LowP never automatically turns the
complete RAG database into a training dataset.

## Optional dependency installation

The Linux and Windows installers offer the optional training dependency group
after a confirmation:

```bash
./install.sh
```

```powershell
.\install.ps1
```

For a development checkout, opt in manually:

```bash
python -m pip install -e ".[training]"
```

This includes PyTorch, Transformers, PEFT, and Accelerate and may download
several GiB. It does not install checkpoints or datasets. QLoRA also requires
bitsandbytes that matches the installed PyTorch and CUDA stack:

```bash
python -m pip install -e ".[qlora]"
```

LowP's automated QLoRA preflight supports NVIDIA GPUs only and requires a
working CUDA PyTorch runtime, `nvidia-smi`, bitsandbytes, and at least 6 GiB of
free VRAM. LoRA uses detected GPU VRAM when CUDA is active; otherwise it applies
a conservative available-RAM check. Disk-space checks include checkpoint size
and checkpoint/adapter output overhead. These are rejection heuristics, not
proof that a run will fit or finish. Architecture, actual activations,
sequence length, package wheels, driver versions, and other running processes
also affect compatibility. CPU training may be very slow.

## Dataset format and explicit import

Supported input files are JSONL/NDJSON, JSON, and CSV and are limited to
256 MiB. JSONL has one conversation/instruction object per line. Accepted
conversation shapes use `messages` or `conversations` arrays with `role` and
`content` fields (`human`/`gpt` role aliases are accepted). Instruction
records use `instruction` or `prompt`, plus `output`, `completion`, or
`response`; optional `input` text is appended to the prompt. CSV uses the same
instruction/output column names. Every example must contain non-empty user and
assistant content.

First inspect and import a specific file. Paths containing spaces should be
quoted:

```text
/train dataset validate "/home/user/training/my examples.jsonl"
/train dataset import "/home/user/training/my examples.jsonl" support-chat
/train dataset list
/train dataset info support-chat
```

Malformed records prevent import. Duplicate normalized examples are counted
and de-duplicated. Imported copies and sidecar metadata are stored under
the LowP data directory (`~/.local/share/lowp/training/datasets/` on Linux,
`%LOCALAPPDATA%\lowp\training\datasets\` on Windows). Selecting a dataset for training is
always explicit.

## Starting and controlling a job

Use a local Hugging Face checkpoint directory and the imported dataset name:

```text
/train start --base "/home/user/models/my-checkpoint" --dataset support-chat
```

Defaults are conservative: LoRA, one epoch, batch size 1, gradient
accumulation 8, sequence length 1024, rank 8, alpha 16, dropout 0.05, and
checkpoint interval 50 steps. Options can be set explicitly:

```text
/train start --base "/home/user/models/my-checkpoint" --dataset support-chat --method lora --epochs 2 --learning-rate 0.0002 --batch-size 1 --gradient-accumulation 8 --max-sequence-length 1024 --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 --checkpoint-steps 50 --output-dir support-adaptation
```

`--output-dir` is a safe name under LowP's managed
`training/outputs/` directory; each job receives its own subdirectory.

For supported NVIDIA QLoRA:

```text
/train start --base "/home/user/models/my-checkpoint" --dataset support-chat --method qlora
```

Check jobs and logs, request cancellation, and resume a job only when it has a
saved checkpoint:

```text
/train status
/train status <job-id>
/train logs <job-id>
/train stop <job-id>
/train resume <job-id>
```

Cancellation is sent to the isolated worker. A worker killed externally is
reported as interrupted when status is next inspected. Only completed
checkpoints can be resumed. Failed runs report an error and are not represented
as successful adapters.

The defaults are configurable per command within bounded values; invalid
values and insufficient detected resources are rejected. Training progress,
metrics, and checkpoint paths are in the job status and `training.log`.

## Adapter management and compatibility

Completed adapter files and metadata live under
the LowP data directory (`~/.local/share/lowp/training/adapters/` on Linux,
`%LOCALAPPDATA%\lowp\training\adapters\` on Windows). They are kept across application
updates and uninstallation by default:

```text
/adapters
/adapters list
/adapters info <adapter-name>
/adapters activate <adapter-name>
/adapters deactivate
/adapters delete <adapter-name>
/adapters delete <adapter-name> confirm
```

The produced PEFT adapters target Hugging Face checkpoint paths. They are not
claimed as active in Ollama chat; activation reports the incompatibility and
leaves the selected model unchanged. This phase does not implement a verified
PEFT-to-Ollama conversion or merge workflow. Deletion requires the explicit
second command and only removes a validated LowP-owned adapter directory.

## Installation lifecycle contract

Every new runtime dependency, optional component, configuration value,
directory, service, and external binary must be integrated into `install.sh`
and `install.ps1`, upgrade/reinstall behavior, health checks, safe
uninstallation, documentation, and tests. Heavy optional components require an explicit choice; hardware-
dependent features must be checked before being offered as usable. Do not
automatically download a model checkpoint or dataset. This rule applies to
future LowP phases as well.
