"""Numbered citations for retrieved internet sources."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class WebCitation:
    label: int
    title: str
    domain: str
    url: str
    published_date: str | None
    accessed_at: datetime
    evidence_type: str
    relevant_excerpt: str = ""
    final_url: str | None = None
    canonical_url: str | None = None

    @property
    def reference(self) -> str:
        fields = [self.domain, self.url]
        if self.final_url and self.final_url != self.url:
            fields.append(f"Retrieved URL: {self.final_url}")
        if self.canonical_url and self.canonical_url not in {
            self.url,
            self.final_url,
        }:
            fields.append(f"Canonical URL: {self.canonical_url}")
        if self.published_date:
            fields.append(f"Published: {self.published_date}")
        fields.append(f"Accessed: {self.accessed_at.date().isoformat()}")
        if self.evidence_type == "search snippet":
            fields.append("Search-result snippet only")
        if self.relevant_excerpt:
            excerpt = " ".join(self.relevant_excerpt.split())
            fields.append(f"Relevant excerpt: {excerpt[:360]}")
        return f"[{self.label}] {self.title} — " + " · ".join(fields)
