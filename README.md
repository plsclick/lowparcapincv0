# LowParLLMCapacityIncV0

LowParLLM is a local AI application that runs directly in a native terminal,
including KDE Konsole, PowerShell, and Windows Terminal. It includes selectable
Ollama models, optional local LoRA/QLoRA training and adapter storage,
installer-managed SearXNG, conservative request routing, local document
retrieval, research, and guarded coding tools.

LowP discovers any model installed in Ollama and keeps the selected model in
the per-user LowP configuration directory (Linux: `~/.config/lowp`; Windows:
`%APPDATA%\lowp`). The existing `qwen3.5:4b` remains the default
unless the user chooses another model. Downloads require explicit confirmation;
LowP never updates or deletes existing models. Conversation history remains in
memory for the current run only and is discarded on exit.

## Status and platform

- Current focus: stabilizing the implemented terminal application and install lifecycle.
- Linux is the verified platform. Windows PowerShell installers and platform
  handling are included, but native Windows release testing remains outstanding.
- Interface: native Python terminal application; no GUI or web frontend.
- Minimum Python: 3.11.
- Ollama must be installed and running with at least one compatible model to
  chat. Startup and status commands remain available without Ollama.

## Development setup

Run from the cloned project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m lowparllm
```

If your shell is Fish, use its matching activation script instead:

```fish
python -m venv .venv
source .venv/bin/activate.fish
pip install -e ".[dev]"
python -m lowparllm
```

Editable installation links the project source into `.venv`; source changes
take effect without reinstalling. The development command does not use or
require `install.sh`.

The terminal shows read-only system and model status at startup. Enter submits
slash commands; ordinary messages can span lines (Enter adds a line, Esc+Enter
submits). Responses stream directly from the local model. Press Ctrl+C during
generation to stop it; `/exit` exits the app. Available commands:

| Command | Action |
|---|---|
| `/help` | List commands |
| `/status` | Show operating system, hardware, Python, Ollama, and model status |
| `/model` | Show the configured model and exact local availability |
| `/model list` / `refresh` | List models currently installed in Ollama |
| `/model info <name>` | Inspect Ollama-reported metadata and capabilities |
| `/model use <name>` | Persistently select an already-installed model |
| `/model pull <name>` | Request a model download; confirm with `confirm` |
| `/train` | Show training and dataset command help |
| `/train hardware` | Show detected hardware and training prerequisites |
| `/train dataset list` | List explicitly imported training datasets |
| `/train dataset validate <path>` | Validate JSONL, JSON, or CSV data |
| `/train dataset import <path> [name]` | Explicitly copy a validated dataset into LowP storage |
| `/train start --base <path> --dataset <name>` | Start a local LoRA job |
| `/train status [job-id]` / `logs` / `stop` / `resume` | Inspect and control training jobs |
| `/adapters [list]` / `info` | Inspect completed LowP PEFT adapters |
| `/adapters activate <name>` | Check activation compatibility (unsupported for Ollama PEFT adapters) |
| `/adapters deactivate` | Report whether a LowP PEFT adapter is active |
| `/adapters delete <name>` | Request explicit confirmation before deleting an adapter |
| `/router` | Show routing settings and the latest decision |
| `/router plan <request>` | Preview a deterministic routing plan |
| `/router capabilities` | Show live capability status and backend limitations |
| `/router auto on` / `off` | Toggle automatic local/web research routing |
| `/generation` | Show generation settings |
| `/generation temperature <0 to 2>` | Set sampling temperature |
| `/generation top_p <0 to 1>` | Set nucleus sampling threshold |
| `/generation top_k <integer>` | Set top-k sampling limit |
| `/generation keep_alive <duration>` | Set how long Ollama keeps the model loaded |
| `/code on` / `/code off` | Switch between coding-agent and normal assistant mode |
| `/code status` / `/code plan <task>` | Inspect coding state or request a read-only plan |
| `/pwd`, `/ls`, `/cd`, `/tree`, `/find` | Navigate the session's active workspace |
| `/project [status|analyze|structure|instructions|open|index]` | Inspect or index the active project |
| `/file read <path>` / `/file diff` | Inspect a project file or the most recent file diff |
| `/search text|file|symbol <query>` | Search project files; plain `/search <query>` searches RAG |
| `/git status|changes|diff|log|branch|review` | Inspect Git state without modifying it |
| `/run <command>` | Queue an allowlisted command for exact-command approval |
| `/test [file|all]`, `/build`, `/lint` | Queue a detected project check for approval |
| `/review [changes|file <path>|security|performance]` | Request a read-only code review |
| `/pdf inspect <path>` | Inspect extractable text and metadata from a local PDF |
| `/pdf extract-text <path> [output.txt]` | Extract text through the existing document-ingestion pipeline |
| `/pdf summarize <path>` | Summarize extracted PDF text using the configured chat model |
| `/doctor` | Check installation, local stores, models, optional services, and paths |
| `/debug <error>` | Investigate an error through the coding workflow |
| `/permissions [status|read-only|workspace-write|ask]` | Inspect or change workspace permissions |
| `/settings` | Show current settings and user data paths |
| `/clear` | Clear the terminal |
| `/reset` | Reset in-memory conversation context |
| `/knowledge` | Show local vector store, registry, and embedding status |
| `/add <path>` | Index one supported document |
| `/add-folder <path>` | Recursively index supported files in a folder |
| `/sources [page]` | List registered documents |
| `/source <id>` | Inspect a document's stored metadata |
| `/search <query>` | Search indexed documents without chatting |
| `/remove-source <id>` | Request confirmation, then remove a source |
| `/remove-all-sources` | Review indexed-source counts before removing all sources |
| `/reindex <id>` | Reprocess one source document |
| `/rag` | Show automatic retrieval status |
| `/rag on` / `/rag off` | Toggle automatic retrieval |
| `/research-local <question>` | Explicitly answer using indexed local documents |
| `/reset-knowledge` | Request confirmation, then remove indexed knowledge |
| `/web` | Show web research provider and status |
| `/web-search <query>` | Search the internet and show source snippets |
| `/web-open <url>` | Safely retrieve and display a public webpage |
| `/research <question>` | Research locally, on the web, or both as appropriate |
| `/research-local <question>` | Use only indexed local documents |
| `/research-web <question>` | Use only internet sources |
| `/research-both <question>` | Combine local and internet sources |
| `/web-save <number>` | Explicitly save a recent web result into local knowledge |
| `/web-settings` | Show web settings; use `enabled`, `auto`, or `save` to toggle |
| `/exit` | Exit cleanly |

Paths containing spaces should be quoted, for example
`/add "/home/user/Study Notes.pdf"`. LowP supports PDF, DOCX, plain text,
Markdown, HTML, and the source formats listed in the
[RAG guide](docs/rag-guide.md). Scanned PDFs without extractable text need OCR,
which is not included.

## Coding workspace

`/code on` enables the project-aware workflow in the current terminal session;
`/code off` returns to the multipurpose assistant without clearing the
conversation. The initial workspace is the launch directory. `/cd` and
`/project open` explicitly change that workspace. Project search and file
operations are bounded to the active directory, skip common generated folders,
and exclude Git-ignored files when Git is available.

The default permission mode is `ask`. Model-proposed edits are shown as diffs
and require `/code continue confirm`; reject a pending plan with `/code stop`.
File writes use a prior content hash to detect concurrent changes. `/run`,
`/test`, `/build`, and `/lint` show the exact command and require `/run confirm`.
Commands use a restricted executable allowlist and do not invoke a shell.
This is conservative application-level checking, **not an OS sandbox**:
approved project code still runs as the current user and can access resources
outside the project. There is no separate network sandbox, and approved project
scripts can make network requests. Review commands and their effects before approval.
`/permissions read-only` disables writes and command execution, while
`/permissions workspace-write` permits direct file operations; agent-proposed
changes still require explicit approval.

This is an initial coding-workspace foundation, not complete Codex parity.
Persistent coding-session history, MCP, language-server symbol analysis,
background process streaming/control, image input, and one-shot `lowp exec`
are not implemented. PEFT adapters remain unavailable to Ollama inference;
web research and RAG continue through their existing LowP commands and router.

Knowledge is stored locally under `~/.local/share/lowp/`, using embedded
Qdrant vectors and a SQLite source registry. Original documents are never
copied, modified, or removed. Folder imports skip hidden and common generated
directories. Chat history remains session-only.

## Ollama and inference settings

The app connects to `http://127.0.0.1:11434` by default, verifies the selected
installed tag before requests, and uses Ollama's `/api/chat` NDJSON stream.
It sends system, user, and successful assistant messages as session context.
Old turns are trimmed as complete user/assistant pairs when the configured
history or approximate character budget is exceeded; `/reset` clears context,
while `/clear` only clears the screen.

On first launch the app creates `~/.config/lowp/config.toml` only if it does
not already exist. Edit that file to customize the base URL, keep-alive,
temperature, top-p, context (`num_ctx`), output limit (`num_predict`), history
limit, and system prompt. `/model use <name>` persists a model switch without
reinstalling LowP or resetting conversation history. The `LOWP_BASE_MODEL`
environment variable, when set, overrides the persisted choice. The default
output limit is `-1` (no fixed prediction limit),
which can take substantial time and memory; it is not infinite generation.
For example:

```toml
[model]
name = "qwen3.5:4b"
provider = "ollama"
base_url = "http://127.0.0.1:11434"
keep_alive = "5m"

[generation]
temperature = 0.7
top_p = 0.8
num_ctx = 4096
num_predict = -1

[conversation]
max_history_messages = 40

[assistant]
system_prompt = "You are a helpful local AI assistant."

[rag]
enabled = true
automatic_retrieval = false
collection_name = "lowp_knowledge"

[rag.embedding]
provider = "ollama"
model = "nomic-embed-text"
batch_size = 16
```

RAG uses the separate Ollama embedding model `nomic-embed-text`; it does not
use or change the selected conversational model. If the embedding model is
missing, LowP explains the exact manual `ollama pull nomic-embed-text` command
and continues to support ordinary chat. LowP never runs that command or
downloads models automatically. See the [RAG guide](docs/rag-guide.md) for
indexing, configuration, and management details.

Internet research uses a self-hosted SearXNG instance, defaulting to
`http://127.0.0.1:8080`; `install.sh` reuses a healthy existing JSON endpoint,
or can create an isolated loopback-only service through an already working
Docker or Podman runtime. LowP has no hosted-provider fallback. Automatic
research is off by default. See the [SearXNG setup guide](docs/searxng-setup.md)
and [web research guide](docs/web-research.md) for configuration, commands,
privacy, caching, and troubleshooting. Pages are never saved to local
knowledge unless `/web-save <number>` is explicitly used.

The router recognizes clear local-document and current-web research requests
using deterministic phrase rules. It does not yet use model-assisted intent
classification, automatically select task-specific models, or activate PEFT
adapters; the current Ollama chat backend is incompatible with the stored
PEFT artifacts. See the [task-routing guide](docs/task-routing.md) for command
behavior and current limits.

Increase `num_ctx` or set `num_predict` to a positive token limit only after
considering the model's supported context and available memory. Unlimited
prediction has no fixed token cap, but still ends when the model or request
stops. `/settings` displays the active values.

Assistant responses use terminal Markdown rendering; website links are
clickable in compatible terminals. Use `/copy` to list generated code blocks
and `/copy 1` to copy a block to the clipboard. Clipboard copying is explicit;
it uses `clip.exe` on Windows or `wl-copy`, `xsel`, or `xclip` on Linux.
Environment overrides include `LOWP_OLLAMA_URL`, `LOWP_TEMPERATURE`,
`LOWP_TOP_P`, `LOWP_CONTEXT_SIZE`, `LOWP_NUM_PREDICT`,
`LOWP_KEEP_ALIVE`, `LOWP_MAX_HISTORY_MESSAGES`, and `LOWP_SYSTEM_PROMPT`.

The app reports generation duration, token counts, and token rate only when
Ollama returns those metrics. It does not expose model thinking fields.

Run tests with:

```bash
python -m pytest
```

Run the Phase 0 environment report with:

```bash
python scripts/check_environment.py
```

See [development setup](docs/development-setup.md) for details.

## Normal user installation

After replacing the repository placeholder with the real GitHub URL:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd LowParLLMCapacityIncV0
./install.sh
lowp
```

For Windows PowerShell 5.1 or PowerShell 7+:

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
Set-Location LowParLLMCapacityIncV0
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
lowp
```

The execution-policy bypass applies only to that PowerShell process. The
installer does not change the machine or user execution policy. It adds the
launcher directory to the current user's PATH; restart PowerShell or Windows
Terminal if `lowp` is not immediately available. If PATH was not updated, add
`%LOCALAPPDATA%\Programs\LowP\bin` to the user PATH in Windows Settings.

The Windows installer creates a separate virtual environment under
`%LOCALAPPDATA%\Programs\LowP\venv`. User configuration is stored under
`%APPDATA%\lowp`; application data and cache use `%LOCALAPPDATA%\lowp`.
Windows paths are not currently native-release-tested; see the
[Windows support guide](docs/windows-support.md).

The Linux installer creates a dedicated virtual environment under
`~/.local/share/lowp/venv/` (or `$XDG_DATA_HOME/lowp/venv/`) and installs the
launcher at `~/.local/bin/lowp`. It does not use the developer's `.venv`, modify
shell startup files, or require root permissions. It installs required
dependencies declared in `pyproject.toml`, offers optional training
dependencies, discovers Ollama models for interactive selection, and asks
before pulling a missing model or SearXNG image. SearXNG deployment requires an
already installed and operational Docker or Podman; the installer never
installs a container runtime. If `~/.local/bin` is not in `PATH`, the installer
prints shell-specific instructions.

Use `lowp doctor` or `lowp --check` to run a non-interactive installation health
check. From the same checkout, `lowp repair` reinstalls the application and required
dependencies, and `lowp update` updates them; both preserve user data and do not
download or replace models.
On Windows, run `.\install.ps1 -Repair` or `.\install.ps1 -Update` from the
checkout, or use `lowp repair` / `lowp update` with the installed launcher.

LoRA training requires a separate local Hugging Face checkpoint directory;
Ollama model tags are inference artifacts and are not directly trainable.
Training packages are optional and installed only after the installer offers
them. No checkpoint or dataset is downloaded automatically. See the
[training guide](docs/training.md) for dataset formats, hardware checks, job
commands, and adapter limitations.

To uninstall the installed runtime and its launcher:

```bash
./uninstall.sh
```

On Windows, run the user-scoped uninstaller:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\uninstall.ps1
```

Both uninstallers require explicit confirmation and preserve user configuration,
models, adapters, training data, knowledge, sessions, and projects. The Windows
uninstaller stops only processes running from LowP's verified virtual
environment and removes only marked application files and its own PATH entry.

The uninstaller asks for confirmation and removes only installation-owned
files. It preserves configuration, persistent user data, datasets, adapters,
the repository, Ollama, and all Ollama models by default. Removing the
LowP-owned SearXNG container and settings requires a separate confirmation;
unrelated containers are not touched.
The uninstaller also stops verified LowP training workers before removing the
runtime; training data and outputs are preserved.
`./uninstall.sh --purge-data` separately offers to remove the explicitly listed
LowP configuration, persistent-data, and cache directories after another
confirmation.

See the [Windows support guide](docs/windows-support.md) for user-level
installation, update, repair, uninstall, PATH, and known validation limits.

## Troubleshooting

- **Missing terminal dependencies:** rerun `./install.sh --repair` from the
  project checkout, or reinstall with `./install.sh`.
- **Python version too old:** use Python 3.11 or newer and recreate the
  environment with that interpreter.
- **`lowp` not found:** add `~/.local/bin` to your shell's `PATH`, then open a
  new terminal or source the relevant shell configuration.
- **Ollama unavailable or model missing:** start Ollama and check `ollama list`.
  Select an installed tag with `/model use <name>` or use `/model pull <name>`
  and then explicitly confirm with `/model pull <name> confirm`.
- **Embedding model missing:** `/knowledge` shows its configured name and the
  manual install command. Installing an embedding model is optional for normal
  chat; LowParLLM never pulls it automatically.
- **Wrong endpoint:** check `model.base_url` in
  `~/.config/lowp/config.toml` on Linux or `%APPDATA%\lowp\config.toml` on
  Windows (or `LOWP_OLLAMA_URL`) and verify that the
  Ollama API is reachable at that address.
- **Invalid config:** edit the per-user `config.toml` (`~/.config/lowp` on
  Linux or `%APPDATA%\lowp` on Windows) and check numeric ranges.
  Context length must be 256–131072; `num_predict` must be `-1` or positive.
- **Installer refuses an existing path:** it found a path it cannot verify as
  owned by LowParLLM. Inspect that path; the installer will not overwrite
  unrelated files.
- **SearXNG is not ready:** verify the local endpoint with
  `http://127.0.0.1:8080/search?q=test&format=json`. If no healthy service is
  present, install/start Docker or Podman as an administrator and rerun
  `./install.sh`; LowP does not install container runtimes.
- **Windows launcher is not found:** restart the terminal; if still absent, add
  `%LOCALAPPDATA%\Programs\LowP\bin` to the current user's PATH and open a new
  terminal.
- **Windows native validation:** this codebase has not yet been run on a Windows
  host. Do not treat the Windows installer and process handling as release
  verified until the native lifecycle tests in the support guide pass.

## Roadmap

Remaining incomplete items include compatible adapter inference and persistent
conversation memory. Image and PDF creation are not part of the current
application.

More information: [architecture](docs/architecture.md),
[model strategy](docs/model-strategy.md), [RAG guide](docs/rag-guide.md),
[task-routing guide](docs/task-routing.md),
[web research](docs/web-research.md).
