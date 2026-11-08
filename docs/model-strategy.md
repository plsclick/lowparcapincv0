# Model strategy and management

## Conversational models

LowP uses Ollama for local conversational inference. The existing
`qwen3.5:4b` model is the initial default, not a hard-coded limitation. LowP
discovers installed models through Ollama's API and reports metadata only when
Ollama supplies it. Unknown context lengths and capabilities are shown as
unknown; no metadata is inferred from model names.

The Linux installer lists models that `ollama list` reports and requires
confirmation before an explicit `ollama pull`; Windows installation only lists
installed models and leaves model selection to `/model`. LowP never updates or
deletes existing models. If Ollama is unavailable, model setup remains deferred.

At any time:

```text
/model
/model list
/model info qwen3.5:4b
/model use qwen3.5:4b
/model refresh
/model pull <model-name>
/model pull <model-name> confirm
```

`/model use` verifies the exact installed tag, writes it to
the per-user config (`~/.config/lowp/config.toml` on Linux,
`%APPDATA%\lowp\config.toml` on Windows), and updates the active Ollama client. Conversation
history is preserved. `LOWP_BASE_MODEL`, if explicitly set in the environment,
continues to override the saved setting. A pull is never treated as a model
switch; use `/model use` after it completes.

Ollama metadata and model compatibility are not guarantees of successful
inference. Context, available RAM/VRAM, quantization, and Ollama runtime
behavior vary. Check `/status` and model details before choosing a larger model.
LowP does not silently select a substitute.

## Training versus Ollama inference

Ollama model tags are runtime artifacts, not necessarily training-ready
checkpoints. LowP's training worker accepts a local Hugging Face model
directory with `config.json` and recognized model weights. It uses
`local_files_only=True` and does not download a checkpoint or run remote model
code. The selected Ollama tag is not assumed to be the matching training base.

LoRA updates a small set of trainable adapter weights while the base checkpoint
is frozen. QLoRA additionally loads a supported NVIDIA checkpoint in 4-bit
quantization. Training is separate from RAG: RAG retrieves selected documents
at inference time; training changes adapter weights based on an explicitly
selected dataset. RAG's complete knowledge store is never automatically used
for training.

The optional `training` dependency group contains PyTorch, Transformers, PEFT,
and Accelerate. The optional `qlora` extra adds bitsandbytes and should only be
installed for a compatible CUDA/PyTorch setup. The Linux and Windows
installers ask before
installing heavyweight training packages. Compatibility still depends on
Python, package wheels, model architecture, driver, GPU memory, system memory,
and disk space. LowP rejects QLoRA when NVIDIA/CUDA prerequisites or its
conservative free-VRAM check fail; this check is not a success guarantee.

## Adapter compatibility

Completed adapters and their metadata are stored under
the per-user LowP data directory (`~/.local/share/lowp/training/adapters/` on
Linux, `%LOCALAPPDATA%\lowp\training\adapters\` on Windows). They remain separate from application
source and survive application updates and uninstallation by default.

PEFT adapters created from a Hugging Face checkpoint are not automatically
loadable by Ollama. LowP does not claim an adapter is active in chat and does
not rewrite the selected Ollama base model. `/adapters activate` validates the
metadata and reports this limitation rather than pretending to activate it.
Conversion or Ollama model packaging requires a separately verified adapter
format and an exact compatible base model; that conversion workflow is not
implemented in this phase.

## Specialised models

The Ollama embedding model remains separate from the conversational model.
Task-specific models, if introduced in a later approved phase, will remain
separate from the conversational base model.
