"""Descriptions and help categories for commands registered by the CLI."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandHelp:
    category: str
    description: str


COMMAND_HELP: dict[str, CommandHelp] = {
    "/help": CommandHelp("General", "Show all available commands"),
    "/status": CommandHelp("General", "Show system and service status"),
    "/settings": CommandHelp("General", "Show current configuration"),
    "/clear": CommandHelp("General", "Clear the terminal"),
    "/reset": CommandHelp("General", "Reset in-memory conversation context"),
    "/exit": CommandHelp("General", "Exit LowP"),
    "/model": CommandHelp("Models and adapters", "Inspect or select installed Ollama models"),
    "/train": CommandHelp("Models and adapters", "Manage optional local LoRA training"),
    "/adapters": CommandHelp("Models and adapters", "Inspect stored PEFT adapters"),
    "/generation": CommandHelp("Models and adapters", "Change sampling and keep-alive settings"),
    "/router": CommandHelp("Routing and capabilities", "Inspect task routing and capability status"),
    "/code": CommandHelp("Coding", "Toggle the project-aware coding workflow"),
    "/permissions": CommandHelp("Coding", "Inspect or change coding permissions"),
    "/review": CommandHelp("Coding", "Review project changes without editing"),
    "/debug": CommandHelp("Coding", "Investigate an error or project file"),
    "/run": CommandHelp("Coding", "Run an approved development command"),
    "/test": CommandHelp("Coding", "Run a detected test command after approval"),
    "/build": CommandHelp("Coding", "Run a detected build command after approval"),
    "/lint": CommandHelp("Coding", "Run a detected lint command after approval"),
    "/init": CommandHelp("Coding", "Create project-specific AGENTS.md guidance"),
    "/pwd": CommandHelp("Project navigation", "Show the active working directory"),
    "/ls": CommandHelp("Project navigation", "List files in the active directory"),
    "/cd": CommandHelp("Project navigation", "Change the active working directory"),
    "/tree": CommandHelp("Project navigation", "Show a bounded directory tree"),
    "/find": CommandHelp("Project navigation", "Find filenames in the active project"),
    "/project": CommandHelp("Project navigation", "Inspect or index the active project"),
    "/file": CommandHelp("Project files and Git", "Read, edit, or diff project files"),
    "/git": CommandHelp("Project files and Git", "Inspect Git status and diffs"),
    "/knowledge": CommandHelp("Local knowledge and RAG", "Show local knowledge-base status"),
    "/add": CommandHelp("Local knowledge and RAG", "Index one supported document"),
    "/add-folder": CommandHelp("Local knowledge and RAG", "Index supported documents in a folder"),
    "/sources": CommandHelp("Local knowledge and RAG", "List indexed document sources"),
    "/source": CommandHelp("Local knowledge and RAG", "Inspect indexed source metadata"),
    "/search": CommandHelp("Local knowledge and RAG", "Search RAG knowledge or project files"),
    "/remove-source": CommandHelp("Local knowledge and RAG", "Remove a source after confirmation"),
    "/remove-all-sources": CommandHelp("Local knowledge and RAG", "Remove indexed sources after confirmation"),
    "/reindex": CommandHelp("Local knowledge and RAG", "Reprocess one indexed source"),
    "/rag": CommandHelp("Local knowledge and RAG", "Show or toggle automatic retrieval"),
    "/reset-knowledge": CommandHelp("Local knowledge and RAG", "Reset indexed knowledge after confirmation"),
    "/web": CommandHelp("Web research", "Show SearXNG research status"),
    "/web-search": CommandHelp("Web research", "Search the configured SearXNG service"),
    "/web-open": CommandHelp("Web research", "Safely retrieve a public webpage"),
    "/research": CommandHelp("Web research", "Research using matching available sources"),
    "/research-local": CommandHelp("Web research", "Research using local documents only"),
    "/research-web": CommandHelp("Web research", "Research using internet sources only"),
    "/research-both": CommandHelp("Web research", "Combine local and internet research"),
    "/web-save": CommandHelp("Web research", "Save a web source into local knowledge"),
    "/web-settings": CommandHelp("Web research", "Show or change research settings"),
    "/web-cache": CommandHelp("Web research", "Show local search-cache statistics"),
    "/web-cache-clear": CommandHelp("Web research", "Clear cached search results"),
    "/web-cache-settings": CommandHelp("Web research", "Change search-cache expiration"),
    "/pdf": CommandHelp("PDF reading", "Inspect, extract text from, or summarize a local PDF"),
    "/doctor": CommandHelp("System and diagnostics", "Check LowP installation and optional services"),
    "/copy": CommandHelp("General", "Copy a generated code block to the clipboard"),
}


def command_help(command: str) -> CommandHelp:
    known = COMMAND_HELP.get(command)
    if known is not None:
        return known
    label = command.lstrip("/").replace("-", " ").replace("_", " ")
    return CommandHelp("Other commands", f"Run the {label} command")
