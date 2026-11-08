# Task routing

LowP has a conservative, deterministic router for natural-language requests
that clearly ask about local documents, current web information, or both.
Those requests use the existing research service and preserve its source
citations. Other messages continue through the configured Ollama chat path.
Automatic local-document retrieval remains a separate setting and is off by
default; greetings and ordinary chat do not search the knowledge base.

The router currently uses explicit phrase rules, not model-assisted
classification. It does not automatically switch conversational models or
activate PEFT adapters. Although LowP can manage stored adapters and optional
training artifacts, the current Ollama chat backend cannot load those PEFT
adapters. Capability inventory and route displays report this limitation
instead of claiming an adapter is active.

## Commands

| Command | Action |
|---|---|
| `/router` or `/router status` | Show router settings and the latest intent |
| `/router plan <request>` | Preview the deterministic plan without executing it |
| `/router explain` | Display the latest routing decision |
| `/router capabilities` | Check implemented capabilities and known backend limits |
| `/router auto on` / `off` | Enable or disable automatic natural-language routing |
| `/router model auto off` | Keep task-specific model selection disabled |
| `/router adapters auto off` | Keep automatic adapter selection disabled |
| `/router reset confirm` | Restore router toggle defaults |

Turning automatic routing off does not disable explicit slash commands such as
`/research-local`, `/research-web`, `/research-both`, `/model`, or `/adapters`.
`/router plan` is a preview only; it does not search, load a model, or change
settings.

Model downloads and training remain explicit operations through the existing
commands and confirmations. LowP does not download a model, checkpoint, or
adapter automatically. `/router capabilities` checks the current Ollama,
RAG, SearXNG, adapter, and optional training status when available.

## Configuration and current scope

Router defaults are stored in `[router]` in
the per-user config (`~/.config/lowp/config.toml` on Linux,
`%APPDATA%\lowp\config.toml` on Windows). The `LOWP_ROUTER_AUTOMATIC`,
`LOWP_ROUTER_AUTO_MODEL`, and `LOWP_ROUTER_AUTO_ADAPTER` environment variables
can override the corresponding toggles. Only the automatic-routing toggle can
currently be enabled as a working automatic behavior; per-task model selection
and adapter activation remain disabled because no compatible implementation
is present.

This phase adds no Python dependencies, background services, autonomous shell
execution, or project-file editing. It reuses the existing RAG, SearXNG,
webpage extraction, caching, model management, and adapter storage features.
The router does not treat retrieved documents or webpages as instructions.
