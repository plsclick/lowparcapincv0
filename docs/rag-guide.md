# Local document knowledge and RAG

LowP can index documents explicitly selected by the user, search their content
semantically, and provide matching excerpts to the existing local chat service.
Documents, extracted metadata, and vectors stay on the local machine. This
feature does not add a web service, cloud database, or alternate conversational
model.

## Requirements and model policy

The selected conversational model is managed separately with `/model`. RAG
uses the separate Ollama embedding model configured by `[rag.embedding].model`, defaulting to
`nomic-embed-text`. Check installed models with `ollama list`. If the embedding
model is absent, LowP reports that indexing and semantic search need it and
shows the exact manual command:

```bash
ollama pull nomic-embed-text
```

Run that command yourself only if you choose to install the model. LowP never
downloads or updates models automatically. Ordinary chat remains available if
the embedding model is unavailable. No document text is sent to an external
embedding service.

## Importing documents

Use `/add` for a file and `/add-folder` for a directory. Quote paths containing
spaces:

```text
/add "/home/user/Study Notes/Computer Networks.pdf"
/add-folder "/home/user/Study Notes"
```

Folder imports are recursive by default. Hidden directories and common
generated directories such as `.git`, `.venv`, `node_modules`, `build`, and
`__pycache__` are skipped. Unsupported file extensions are ignored. Each
supported document is processed independently; an individual extraction or
indexing failure is reported without stopping the remaining import. The
default file-size limit is 50 MiB.

Supported formats:

| Format | Extensions |
|---|---|
| PDF | `.pdf` |
| Plain text | `.txt` |
| Markdown | `.md` |
| Word documents | `.docx` |
| Source and data files | `.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`, `.json`, `.html`, `.css` |

PDF text and page numbers are extracted page by page. Encrypted or scanned
PDFs without extractable text are reported; OCR is not provided. DOCX
paragraphs, headings, and tables are extracted. Text and source-code
indentation and line breaks are preserved where possible. Original files are
never copied, modified, executed, or deleted.

LowP indexes only the exact files or folders given to these commands. It does
not scan the computer or watch for future changes. Re-run `/add` on an updated
file or use `/reindex <id>` to refresh an existing source. Unchanged content is
deduplicated by file hash, including files moved or renamed to a new path.

## Search and chat

Use `/search <query>` to retrieve matching passages without sending the query
to the chat model. Results show the source reference, excerpt, and actual
Qdrant similarity score. Use `/source <id>` to inspect stored source metadata.

Automatic retrieval is off by default. Ordinary chat messages, including
greetings, do not search indexed documents or add their content to the model
prompt. To use local knowledge for a question, explicitly run
`/research-local <question>` or `/search <query>`. You can opt in to automatic
retrieval with `/rag on`; when enabled, LowP searches local knowledge before
chat turns and adds relevant excerpts with citations. `/rag off` disables it
again and persists that choice.

The configured similarity threshold filters results; a passage below the
threshold is not presented as a match. If nothing qualifies, chat proceeds
without document context. Citations are generated only from retrieved source
metadata. PDFs include actual page numbers; other sources include headings or
line ranges only where available.

Retrieved documents are untrusted reference material, not instructions. LowP
keeps that context separate from system instructions and never executes code,
commands, macros, or URLs found in documents. Prompt injection cannot be
eliminated completely, so review document-supported answers and citations.

## Managing sources

```text
/knowledge
/sources
/sources 2
/source <id>
/search <query>
/reindex <id>
/remove-source <id>
/remove-source <id> confirm
/remove-all-sources
/remove-all-sources confirm
/rag status
/rag off
/rag on
/reset-knowledge
/reset-knowledge confirm
```

`/sources` is paginated, with 20 records per page. The ID prefix shown there
can be used with `/source`, `/reindex`, and `/remove-source`. Removal requires
the explicit `confirm` argument and deletes only LowP's metadata and vectors;
it never deletes the original file. Reset likewise requires confirmation and
removes LowP's source registry and vector collection, not source documents,
chat context, or other application data.

To clear every indexed source at once, use `/remove-all-sources`, review the
displayed source and chunk counts, then enter `/remove-all-sources confirm`.
This removes LowP's indexed source records and vectors only; original files,
chat context, and other application data remain untouched.

`/rag off` disables automatic retrieval and persists the setting in the user
configuration. `/rag on` enables it again. `/search` remains a manual search
operation, while `/research-local <question>` explicitly searches local
knowledge and prepares a cited answer. `/knowledge` reports database
availability, source and chunk counts, embedding-model availability,
collection name, storage locations, and the automatic retrieval setting.

## Configuration and storage

RAG settings live in the per-user LowP configuration. Linux defaults to
`~/.config/lowp/config.toml` (or configured XDG paths); Windows uses
`%APPDATA%\lowp\config.toml`. Existing configuration is preserved; missing RAG
values use safe defaults.

```toml
[rag]
enabled = true
automatic_retrieval = false
collection_name = "lowp_knowledge"

[rag.embedding]
provider = "ollama"
model = "nomic-embed-text"
batch_size = 16

[rag.chunking]
chunk_size = 800
chunk_overlap = 120

[rag.retrieval]
top_k = 5
score_threshold = 0.35
max_context_tokens = 3000

[rag.storage]
qdrant_path = "~/.local/share/lowp/qdrant"
sqlite_path = "~/.local/share/lowp/knowledge.db"

[rag.ingestion]
max_file_size_mb = 50
recursive = true
ignore_hidden = true
exclude_patterns = [".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist", ".tox", ".mypy_cache", ".pytest_cache", ".next", "target"]
```

Chunk size and overlap are word-budget based, not model-token guarantees.
Defaults are validated and can be changed to fit the selected embedding model
and available hardware. The Qdrant collection records and checks the embedding
model and vector dimensions; do not point different embedding models at the
same populated collection.

The storage paths shown above are Linux examples. Persistent knowledge is
stored locally in the configured Qdrant directory and SQLite registry, by
default under Linux `~/.local/share/lowp/` or Windows
`%LOCALAPPDATA%\lowp\`. It is independent of the project checkout, `.venv`, and
original documents. The normal installer and uninstaller preserve this data.
Back up these files while LowP is closed.
