# Architecture

## Native terminal application

LowParLLM is a Python application intended to run directly in Linux terminals
such as KDE Konsole and Windows terminals such as PowerShell and Windows
Terminal. It has no graphical frontend, browser interface, embedded terminal,
or web service. The terminal loop uses `asyncio`, Rich for rendering, and
Prompt Toolkit for multiline input. Linux is verified; native Windows
release-testing remains outstanding.

## Application and capability boundaries

`lowparllm.core` contains shared configuration, path, and logging foundations.
The terminal interface lives in `lowparllm.cli`; AI capabilities live
under `lowparllm.ai`; and application workflows can use `lowparllm.services`.
This keeps presentation, orchestration, and capability-specific code separate
without introducing a service framework or unnecessary abstractions.

## Local inference

Ollama is the local inference boundary. `qwen3.5:4b` is the initial default;
the model manager discovers available installed Ollama models, reports
Ollama-provided metadata, and persists a user-verified selection. The async
HTTPX client checks the exact local tag, then consumes Ollama's `/api/chat`
newline-delimited JSON stream.
The renderer receives user-facing content events without handling HTTP. The
client never silently pulls or substitutes models and ignores model thinking
metadata. Explicit model switching changes the client for subsequent
inference without clearing conversation history. Stream cleanup closes the
HTTP response on completion, failure, or cancellation. Temperature, top-p,
context size, output limit, and keep-alive
are validated configuration values passed to Ollama.

## Storage

Application configuration, persistent user data, and cache are kept separate
using platform user directories: Linux XDG defaults (`~/.config/lowp`,
`~/.local/share/lowp`, `~/.cache/lowp`) and Windows AppData directories
(`%APPDATA%\lowp`, `%LOCALAPPDATA%\lowp`, `%LOCALAPPDATA%\lowp\cache`). Source
code remains independent of user data. Embedded Qdrant vectors live under the
persistent data directory; SQLite source metadata is stored in
`knowledge.db`. One managed Qdrant client is used per application process.
Ollama's `/api/embed` generates local embeddings, and stored model and
dimension metadata prevent incompatible vectors from being mixed.

## Local document retrieval

The RAG pipeline is separated from terminal rendering and conversational
inference: explicit file selection → format-specific extraction → bounded,
metadata-preserving chunks → Ollama embeddings → persistent Qdrant and SQLite
→ semantic retrieval → ephemeral untrusted context for the existing chat
service → source citations. Retrieved context is not committed to conversation
history, and retrieved text cannot replace system instructions. Unsupported
files and scanned PDFs without extractable text are reported; OCR is not
implemented. Folder traversal skips hidden/generated directories and LowP's
own data directory. Original files are never modified.

The selected conversational model is managed separately. The default embedding
model is `nomic-embed-text`; it is never downloaded
automatically. If it is absent, indexing and RAG search explain how to install
it manually, while normal chat remains available. See [RAG guide](rag-guide.md)
for commands, settings, and data management.

## Internet research

The optional web-search layer consists of a SearXNG HTTP JSON client, a
separate SQLite search-result cache, a bounded webpage fetcher, a standard
library HTML article extractor, URL/DNS validation, and research orchestration.
The research workflow asks the local Qwen model for focused queries without
including chat history or local document contents, searches asynchronously
within configured limits, removes duplicate URLs, retrieves public webpages,
and supplies bounded untrusted excerpts to Qwen for the answer. Citations show
the source title, domain, URL, publication date when known, access date, and a
relevant excerpt. Search snippets are explicitly distinguished from extracted
page content.

Automatic web research is disabled by default and can be enabled explicitly.
The CLI also supports local-only, web-only, and combined research. Internet
content is never indexed automatically; `/web-save` copies a selected source
into LowP's managed data directory and sends it through the existing RAG
ingestion pipeline. SearXNG is the sole search provider, configured through
`SEARXNG_BASE_URL` or `[web].searxng_base_url`. Local-only research does not
access SearXNG, and web research does not send local documents or conversation
history to the search instance.

Webpage access is HTTP(S)-only, public-address-only, redirect-validated, size
bounded, and performed without executing JavaScript. robots.txt must permit
access; policy-check failures stop retrieval. See [web research guide](web-research.md).

Chat messages live in memory for the current run only. Complete older turns
are trimmed based on a message cap and approximate character budget derived
from configured context size, keeping recent complete turns. `/reset` clears
this in-memory history; `/clear` only clears the terminal. The default system
prompt and inference settings live in the per-user config
(`~/.config/lowp/config.toml` on Linux,
`%APPDATA%\lowp\config.toml` on Windows), created only if absent. The installed
virtual environment is kept separately from user data.

## Project-aware coding workflow

`lowparllm.ai.coding` supplies a session-scoped workspace, bounded project
discovery and file operations, an explicitly approved shell-free command
runner, and a small tool loop that reuses the existing `ChatService`. On
Windows, a narrow read-only PowerShell commandlet subset is translated into
fixed PowerShell expressions; arbitrary script execution remains blocked.
`/code on` routes ordinary chat tasks through that loop; global commands such
as `/model`, RAG, and research remain handled by the existing command and
service paths. Project searches use Git's `--exclude-standard` listing when
available, so untracked ignored files are excluded. File operations are
restricted to the active workspace.

The permission default is `ask`. Changes are staged with a diff and require
explicit confirmation. File replacement checks the SHA-256 observed before
approval. Commands require separate confirmation, use an executable allowlist,
and are started without a shell. These checks are application-level
restrictions, not an OS sandbox: an approved child process runs with the
user's operating-system privileges. No separate network sandbox is provided;
approved project scripts may access the network.

The coding workflow does not claim unsupported behavior: there is no
persistent coding-session store, MCP integration, language-server symbol
index, image input, one-shot `lowp exec`, or interactive background-process
manager. PDF output generation is not implemented; PDF input extraction,
summarization, and RAG ingestion remain supported through the same safe
document pipeline. Invalid model-generated action plans fail visibly rather
than being executed. Existing PEFT metadata and training do not make adapters
active in the Ollama chat backend.

## Local training and adapters

The optional `lowparllm.ai.lora` subsystem validates explicitly imported
datasets, checks local Hugging Face checkpoint files and available machine
resources, and launches a separate Python worker for Transformers/PEFT LoRA or
supported NVIDIA QLoRA jobs. The worker uses local-only model loading, writes
logs and resumable checkpoints under the LowP data directory, and records
metadata for completed adapters. It does not import the RAG knowledge base or
download training checkpoints.

Ollama inference models and Hugging Face training checkpoints are separate
artifacts. PEFT adapters are not advertised as active in Ollama chat; this
release does not implement a verified conversion/packaging flow.

## Document ingestion

`lowparllm.ai.rag.ingestion` reads PDF, DOCX, Markdown, plain-text, and supported
source files for indexing and retrieval. PDF text and page metadata are
extracted through `pypdf`; scanned PDFs without extractable text require an
external OCR workflow. `/pdf inspect`, `/pdf extract-text`, and `/pdf summarize`
use this same extraction pipeline. LowP does not create or convert PDFs.

## Installation lifecycle

`install.sh` owns the dedicated application virtual environment and launcher.
It installs required dependencies from `pyproject.toml`, offers optional
training dependencies, model selection, and a
loopback-published SearXNG container when an operational Docker or Podman
runtime is already available. SearXNG uses a LowP-owned settings directory and
container label. Existing healthy JSON endpoints are reused; unowned
containers and configuration are not overwritten. `uninstall.sh` preserves
user data and requires a separate confirmation to remove only marked LowP
SearXNG resources.

The installed launcher exposes `lowp --check` for required-component health,
`lowp repair` to reinstall the application and its declared dependencies from
the original checkout, and `lowp update` for the same non-destructive
application update workflow. Both maintenance commands leave user data and
external models untouched.

User data remains separate from the marked application runtime. See [training](training.md),
[SearXNG setup](searxng-setup.md), and the [model strategy](model-strategy.md).

## Current limits

Image and PDF creation are not implemented. Coding assistance is an initial,
deliberately bounded workflow; see the project workspace section above and the
limitations in the README.

## Future Windows compatibility

Linux-specific environment inspection is isolated in
`lowparllm.services.system_info`, while application user paths use the dedicated
`core.paths` module and standard Python path utilities. Future Windows support
can add platform-specific path and environment handling behind these
boundaries; Windows is not supported yet.
