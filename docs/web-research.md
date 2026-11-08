# Web research

LowP searches only through an explicitly configured SearXNG instance. The
default endpoint is `http://127.0.0.1:8080`; LowP never falls back to a hosted
search provider. SearXNG setup, JSON configuration, and container commands are
covered in [the SearXNG setup guide](searxng-setup.md).

## Connection and settings

Configure the endpoint in the per-user config (`~/.config/lowp/config.toml` on
Linux, `%APPDATA%\lowp\config.toml` on Windows) under `[web]`, or set
`SEARXNG_BASE_URL` in the shell that launches LowP:

```bash
export SEARXNG_BASE_URL=http://127.0.0.1:8080
lowp
```

Plain HTTP is allowed only for a loopback address. Explicitly configured remote
instances must use HTTPS. The endpoint must be trusted by the user; remote
SearXNG operators can observe submitted queries.

Use `/web` to check availability. A stopped or unreachable instance is
reported as unavailable; search does not create substitute results. `/web`
also displays the configured endpoint and connection status. Automatic
research is off by default.

Settings can be changed in the terminal and persist to the LowP configuration:

```text
/web-settings auto on
/web-settings enabled off
/web-settings base_url http://127.0.0.1:8080
/web-settings language en
/web-settings categories general,science
/web-settings safe_search 1
/web-settings max_queries 3
/web-settings max_results 5
/web-settings max_pages 5
/web-settings timeout_seconds 20
/web-settings max_concurrent_requests 3
/web-settings search_cache_ttl_seconds 900
/web-settings unlimited on
```

Supported safe-search levels are 0 (off), 1 (moderate), and 2 (strict).
Search categories and language identifiers are passed to the configured
instance, which determines the available engines and languages. The unlimited
toggle removes configured per-task quotas while retaining finite safety limits
for pagination, page retrieval, concurrency, timeouts, response size, and model
context.

Search result sets are cached in a separate SQLite database under LowP's
user-cache directory, not in the permanent RAG knowledge database. Inspect,
clear, or change expiration with:

```text
/web-cache
/web-cache-clear
/web-cache-settings
/web-cache-settings ttl 900
/web-search --fresh latest Python release
/research-web --fresh current Python release
```

Fresh requests bypass cache reads and update the cache after a successful
response. A TTL of zero disables cache use.

## Commands and research scope

```text
/web-search latest Python release
/web-open https://docs.python.org/3/
/research compare current Python packaging tools
/research-local summarize my indexed project notes
/research-web latest official Python release information
/research-both compare my notes with current official documentation
/web-save 1
```

`/research <question>` chooses local-only mode when the question clearly refers
to personal or indexed material, combined mode when it also asks for current
information, and web-only mode otherwise. Explicit mode words after
`/research` and the dedicated research commands select local, web, or combined
sources. `/research-local` never accesses SearXNG; `/research-web` does not
search local RAG; `/research-both` uses both and distinguishes their citations.

The web research pipeline asks the local Qwen3.5 4B model to plan focused
queries, searches SearXNG, deduplicates results, retrieves selected public
pages, extracts bounded readable content, and generates a cited response with
the local model. `/web-search` lists snippets and source metadata. `/web-open`
extracts a single page. `/web-save <number>` explicitly saves a selected
source through the existing RAG ingestion path; pages are never indexed
automatically.

Assistant responses render Markdown in the terminal. HTTP(S) links in the
response, citations, and search results are clickable in terminals that support
OSC 8 hyperlinks, including KDE Konsole. Fenced code blocks remain selectable
with the terminal and can also be copied explicitly:

```text
/copy
/copy 1
```

`/copy` lists the code blocks in the latest assistant response. `/copy 1`
copies its first fenced block to the system clipboard using `wl-copy`, `xsel`,
or `xclip` when available. If no clipboard helper is installed, select the
block and press Ctrl+Shift+C in Konsole. LowP does not copy response content
automatically.

## Privacy and security

- LowP submits only the current question and query terms to the configured
  SearXNG endpoint. Conversation history and local RAG documents are not sent
  during web-only research.
- A remote endpoint receives queries and connection metadata. Use an endpoint
  you trust, use HTTPS, and do not send sensitive questions to a third party.
- SearXNG aggregates external search engines; rate limits, engine availability,
  and site access rules still apply. LowP does not try to evade them.
- Retrieved pages are untrusted data. Page instructions cannot override
  application rules; LowP does not execute webpage commands or code.
- Webpage retrieval uses the existing HTTPX-based extractor, does not execute
  JavaScript, and does not download arbitrary files. Public URL validation,
  redirect validation, response limits, and robots policy are enforced.
- Search results are cached locally with user-only file permissions and are
  separate from permanent knowledge. `/web-cache-clear` removes the cache.
- Search and webpage results are not automatically saved to RAG.

## Troubleshooting

- **SearXNG unavailable:** run `/web`, check that the instance is running, and
  verify the configured base URL. LowP requires a reachable SearXNG instance
  for web commands and does not use cloud fallback.
- **HTTP 403 on JSON search:** enable `json` under SearXNG's
  `search.formats` configuration and restart the instance.
- **No results:** inspect SearXNG's configured engines, categories, language,
  and safe-search setting. An empty response is reported as empty, not
  replaced with fabricated results.
- **Webpages are not extracted:** pages may reject requests, robots policy may
  disallow retrieval, or URL/redirect safety validation may reject the host.
- **Research answer is unavailable:** Qwen3.5 4B and Ollama are required for
  query planning and answer generation. `/web-search` can independently test
  the SearXNG JSON endpoint.
- **A saved source is not indexed:** check the configured embedding service and
  RAG status. LowP does not download models automatically.

Automated tests mock SearXNG, DNS, HTTP responses, and pages; they do not
require a live internet connection. Live SearXNG connectivity is not asserted
by the normal test suite.
