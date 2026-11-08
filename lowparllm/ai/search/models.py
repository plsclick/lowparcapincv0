"""Typed values shared by web-search providers and research workflows."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    domain: str
    published_date: str | None = None
    score: float | None = None
    engine: str | None = None
    category: str | None = None
    position: int | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WebPage:
    title: str
    url: str
    canonical_url: str | None
    domain: str
    published_date: str | None
    content: str
    accessed_at: datetime


@dataclass(frozen=True)
class WebSource:
    source_id: str
    title: str
    url: str
    domain: str
    published_date: str | None
    accessed_at: datetime
    content: str
    evidence_type: str
    fetch_error: str | None = None
    final_url: str | None = None
    canonical_url: str | None = None
