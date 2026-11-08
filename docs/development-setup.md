# Development setup

## Prerequisites

- Linux for verified development; Windows support code is present but not yet
  validated on a native Windows host (see [Windows support](windows-support.md))
- Python 3.11 or newer with `venv` and `pip`
- Git (recommended)
- Ollama running locally with at least one installed chat-compatible model
- Ollama embedding model `nomic-embed-text` for document indexing and semantic search (optional)

Ollama and its models are not changed by the development setup. The status UI
and commands remain usable when Ollama is unavailable.

## Development environment

From any selected checkout location, run these commands at the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m lowparllm
```

For Windows PowerShell:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\python.exe -m lowparllm
```

Using the environment's executable directly avoids relying on the
PowerShell script activation policy.

For Fish, activate the environment with the Fish-specific script:

```fish
python -m venv .venv
source .venv/bin/activate.fish
pip install -e ".[dev]"
python -m lowparllm
```

The editable install keeps `.venv` inside the repository and reflects source
changes without reinstalling. `python -m lowparllm` works from the repository
root and does not require `install.sh` or a global `lowp` command.
The editable project install includes the HTTPX, embedded Qdrant, PDF, and
DOCX dependencies needed by the current application. After pulling project
changes that update dependencies, rerun `pip install -e ".[dev]"`.
Phase 4 also uses HTTP Core for its DNS-pinned HTTP transport.

Optional training dependencies can be installed explicitly:

```bash
python -m pip install -e ".[training]"
```

This may download large packages. It does not install a model checkpoint or
dataset. QLoRA additionally requires an appropriate CUDA-enabled PyTorch setup
and optional bitsandbytes:

```bash
python -m pip install -e ".[qlora]"
```

Use the installer prompts for normal user installations. The `qlora` extra
alone is insufficient; install and verify the `training` extra and CUDA stack
first. See [training](training.md) for compatibility limits.

Run automated tests:

```bash
python -m pytest
```

## User-level installation

For Linux end users, run `./install.sh` from the project checkout. It installs the
application into a separate environment under
`~/.local/share/lowp/venv/` (or `$XDG_DATA_HOME/lowp/venv/`) and creates
`~/.local/bin/lowp`. It requires no root privileges and does not touch the
developer `.venv`. If the launcher directory is not in `PATH`, the script
prints guidance without editing shell startup files. Training dependencies are
optional and separately confirmed; ordinary
installation always installs the application's required dependencies from
`pyproject.toml`.

For Windows, run `.\install.ps1` from PowerShell using a process-scoped
execution-policy bypass as documented in [Windows support](windows-support.md).
The Windows installer creates a per-user environment and adds only its launcher
directory to the user PATH.

Run `./uninstall.sh` on Linux or `.\uninstall.ps1` on Windows to remove the owned installed environment and launcher
after confirming. It preserves user configuration, data, datasets, and
adapters, and stops verified active LowP training workers before removing the
runtime. A separate confirmation is required to remove the LowP-owned
SearXNG container and settings. `./uninstall.sh --purge-data` offers an
additional, separately confirmed removal of the listed LowP configuration,
persistent-data, and cache directories.

The installed `lowp doctor` (or `lowp --check`) command reports required installation health.
`lowp repair` and `lowp update` rerun the installer from its original checkout;
they reinstall the application and required dependencies without resetting
configuration, models, or persistent data.

## Environment report

```bash
python scripts/check_environment.py
```

The detector uses Python's standard library and optional read-only system
utilities. It reports absent hardware/software gracefully and never downloads
models or changes system configuration.

## Ollama and conversation configuration

On first application launch, `~/.config/lowp/config.toml` is created only when
absent. It can configure the endpoint, temperature, top-p, context (`num_ctx`),
output limit (`num_predict`), keep-alive duration, session history cap, and
system prompt. The initial conversational default is `qwen3.5:4b`. Installed
Ollama models can be listed and selected with `/model`; the selection persists
in the config file. An explicit `LOWP_BASE_MODEL` environment override takes
precedence. The default context is 4096 tokens. `num_predict = -1` means no
fixed output limit and may use significant time and memory; it is not infinite
generation.

Chat responses stream directly from Ollama. Ctrl+C cancels generation and
returns to the prompt. Conversation context exists only in memory until exit;
`/reset` clears it while `/clear` only clears the terminal.

For long generations, set `num_predict = -1` (the default) for no fixed output
limit, or use a positive value to cap generated tokens. This is not infinite
generation and may use significant time and memory. Set `num_ctx` to configure
the requested context window; use values supported by the model and hardware.
Other generation values are shown by `/settings`.

Use `/generation` to view and change generation controls while LowP is
running:

```text
/generation temperature 0.7
/generation top_p 0.8
/generation top_k 20
/generation keep_alive 5m
```

Temperature accepts 0 to 2, top-p accepts 0 to 1, and top-k must be a positive
integer. Keep-alive must be a non-empty Ollama duration such as `5m`, `1h`,
`0`, or `-1`. Each successful change takes effect for subsequent requests in
the current session and is persisted in the matching `[generation]` or
`[model]` key in `~/.config/lowp/config.toml`. Existing environment overrides
(`LOWP_TEMPERATURE`, `LOWP_TOP_P`, `LOWP_TOP_K`, and `LOWP_KEEP_ALIVE`) still
take precedence when LowP starts again.

## Project-aware coding workflow

Start LowP in the directory you want to inspect, then use `/code on`. The
active path appears in the prompt and can be changed with `/cd "path with
spaces"` or `/project open "path with spaces"`. Use `/project status` to see
detected manifests and Git state. `/search text <query>` searches project
files; plain `/search <query>` continues to search indexed RAG knowledge.

The default permission mode is `ask`. Proposed agent edits are previewed and
need `/code continue confirm`; development commands need `/run confirm`.
`/code off` returns to ordinary assistant mode. Path validation is not OS-level
isolation: approved commands execute as the current Linux user and may access
resources outside the workspace or make network requests. Review commands
before approval. Persistent
coding sessions, MCP, image context, and one-shot `lowp exec` are not currently
provided.

## Internet research

Web search requires a self-hosted SearXNG instance. `./install.sh` reuses a
healthy JSON endpoint or offers to deploy a pinned image through an already
operational Docker or Podman runtime. It does not install those runtimes.
LowP defaults to
`http://127.0.0.1:8080` and does not fall back to a hosted provider. Configure
`SEARXNG_BASE_URL` in the environment or set the `[web].searxng_base_url` key
in `~/.config/lowp/config.toml`. See the
[SearXNG setup guide](searxng-setup.md) for Linux container and JSON setup.

Automatic web research is off by default. Inspect and toggle it with:

```text
/web
/web-settings auto on
/web-settings auto off
```

See the [web research guide](web-research.md) for commands, privacy behavior,
network protections, and troubleshooting.

To update an end-user installation after pulling source changes, run
`./install.sh` again. It updates the installed environment without replacing
user configuration or application data.
