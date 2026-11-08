"""Small standard-library HTML article extractor with no JavaScript execution."""

from datetime import datetime
from html.parser import HTMLParser
import re
from urllib.parse import urljoin, urlsplit

from lowparllm.ai.search.models import WebPage

_BLOCKED = {
    "script", "style", "noscript", "iframe", "svg", "canvas", "nav", "footer",
    "header", "aside", "form", "button", "select", "template", "dialog",
}
_TEXT_BLOCKS = {"h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre", "td"}
_VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
_META_DATE_KEYS = {
    "article:published_time",
    "date",
    "datepublished",
    "datePublished",
    "og:published_time",
    "pubdate",
    "publish_date",
    "parsely-pub-date",
}
_NOISE = re.compile(
    r"(?:comment|sidebar|share|related|breadcrumb|advert|promo|cookie|banner|social|menu)",
    re.I,
)
_MAIN = re.compile(r"(?:article|content|main|post|story|entry)", re.I)


class _ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self._in_title = False
        self._skip_stack: list[str] = []
        self._main_depth = 0
        self._block_stack: list[tuple[str, bool, int, list[str]]] = []
        self._main_tag_stack: list[str] = []
        self.blocks: list[tuple[int, str]] = []
        self.metadata: dict[str, str] = {}
        self.canonical: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        classes = f"{values.get('id', '')} {values.get('class', '')}"
        if tag == "meta":
            key = values.get("property", "") or values.get("name", "")
            value = values.get("content", "").strip()
            if key.lower() in {entry.lower() for entry in _META_DATE_KEYS} and value:
                self.metadata.setdefault("published_date", value)
            elif key.lower() in {"og:title", "twitter:title"} and value:
                self.metadata.setdefault("meta_title", value)
            return
        if tag == "link" and "canonical" in values.get("rel", "").lower().split():
            self.canonical = values.get("href", "").strip() or None
            return

        if self._skip_stack:
            if tag not in _VOID:
                self._skip_stack.append(tag)
            return
        if tag in _BLOCKED or (tag not in _TEXT_BLOCKS and _NOISE.search(classes)):
            self._skip_stack.append(tag)
            return
        if tag in {"main", "article"} or _MAIN.search(classes):
            self._main_depth += 1
            self._main_tag_stack.append(tag)
        if tag == "title":
            self._in_title = True
        if tag in _TEXT_BLOCKS:
            noise = bool(_NOISE.search(classes))
            boost = 4 if self._main_depth else 1
            if _NOISE.search(classes):
                boost = 0
            elif _MAIN.search(classes):
                boost += 1
            boost = 0 if noise else 4 if self._main_depth else 1
            self._block_stack.append((tag, not noise, boost, []))

    def handle_endtag(self, tag: str) -> None:
        if self._skip_stack:
            if tag == self._skip_stack[-1]:
                self._skip_stack.pop()
            return
        if tag == "title":
            self._in_title = False
        if self._block_stack and tag == self._block_stack[-1][0]:
            _, include, weight, parts = self._block_stack.pop()
            text = _normalize("".join(parts))
            if include and text:
                self.blocks.append((weight, text))
        if tag in self._main_tag_stack:
            index = len(self._main_tag_stack) - 1 - self._main_tag_stack[::-1].index(tag)
            del self._main_tag_stack[index]
            self._main_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        if self._in_title:
            self.title_parts.append(data)
        self._append_text(data)

    def handle_entityref(self, name: str) -> None:
        self.handle_data(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.handle_data(f"&#{name};")

    def _append_text(self, text: str) -> None:
        if self._block_stack:
            self._block_stack[-1][3].append(text)


def extract_article(
    html: str,
    *,
    url: str,
    accessed_at: datetime,
    max_chars: int,
) -> WebPage:
    """Extract the best readable content and factual metadata from HTML."""
    parser = _ArticleParser()
    try:
        parser.feed(html)
        parser.close()
    except (ValueError, AssertionError) as exc:
        raise ValueError("The webpage contains malformed HTML that could not be parsed.") from exc

    ranked = sorted(
        enumerate(parser.blocks),
        key=lambda item: (item[1][0], len(item[1][1].split())),
        reverse=True,
    )
    if not ranked:
        raise ValueError("No readable article text was found on the webpage.")
    if any(weight >= 4 for _, (weight, _) in ranked):
        chosen = [block for _, (weight, block) in sorted(ranked) if weight >= 4]
    else:
        chosen = [block for _, (_, block) in sorted(ranked)]
    content = "\n\n".join(dict.fromkeys(chosen)).strip()
    if not content:
        raise ValueError("No readable article text was found on the webpage.")
    content = content[:max_chars]

    title = _normalize("".join(parser.title_parts))
    title = title or parser.metadata.get("meta_title") or ""
    parsed_url = urlsplit(url)
    if not title:
        title = parsed_url.hostname or url
    canonical = None
    if parser.canonical:
        from lowparllm.ai.search.url_security import normalize_public_url

        try:
            candidate = normalize_public_url(urljoin(url, parser.canonical))
            if urlsplit(candidate).hostname == parsed_url.hostname:
                canonical = candidate
        except ValueError:
            pass

    page_date = parser.metadata.get("published_date")
    return WebPage(
        title=title[:300],
        url=url,
        canonical_url=canonical,
        domain=parsed_url.hostname or "",
        published_date=page_date,
        content=content,
        accessed_at=accessed_at,
    )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
