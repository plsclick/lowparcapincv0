"""Bounded public-webpage fetching with redirect and robots policy checks."""

import asyncio
from datetime import datetime, timezone
from email.message import Message
from collections.abc import Callable
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from lowparllm.ai.search.html_extractor import extract_article
from lowparllm.ai.search.models import WebPage
from lowparllm.ai.search.url_security import (
    PinnedTransport,
    URLSafetyError,
    normalize_public_url,
    resolve_public_addresses,
)

USER_AGENT = "LowPResearchBot/1.0"
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_MAX_REDIRECTS = 5
_ROBOTS_LIMIT = 256 * 1024


class WebFetchError(RuntimeError):
    """A public webpage could not be safely fetched or extracted."""


class WebPageFetcher:
    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        max_page_bytes: int = 2 * 1024 * 1024,
        max_page_chars: int = 12000,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_page_bytes = max_page_bytes
        self.max_page_chars = max_page_chars
        self._client = client
        self._robots: dict[str, RobotFileParser] = {}
        self._robots_locks: dict[str, asyncio.Lock] = {}

    async def fetch(
        self,
        value: str,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> WebPage:
        try:
            url = normalize_public_url(value)
            await self._check_robots(url)
            final_url, headers, body = await self._request(url, self.max_page_bytes)
            content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise WebFetchError(
                    f"The page uses unsupported content type {content_type or '(missing)'!r}; "
                    "only HTML pages are opened."
                )
            text = _decode_html(body, headers.get("content-type", ""))
            if progress is not None:
                progress("Extracting webpage content")
            return extract_article(
                text,
                url=final_url,
                accessed_at=datetime.now(timezone.utc),
                max_chars=self.max_page_chars,
            )
        except URLSafetyError as exc:
            raise WebFetchError(str(exc)) from exc
        except WebFetchError:
            raise
        except (httpx.HTTPError, OSError, UnicodeError, ValueError) as exc:
            raise WebFetchError("The webpage could not be retrieved or parsed.") from exc

    async def _check_robots(self, url: str) -> None:
        parsed = urlsplit(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin in self._robots:
            rules = self._robots[origin]
            if not rules.can_fetch(USER_AGENT, url):
                raise WebFetchError("This website disallows retrieval in its robots.txt.")
            return
        lock = self._robots_locks.setdefault(origin, asyncio.Lock())
        async with lock:
            if origin not in self._robots:
                robots_url = urljoin(origin, "/robots.txt")
                try:
                    _, _, body, status = await self._request_status(
                        robots_url, _ROBOTS_LIMIT, check_robots=False
                    )
                except (WebFetchError, URLSafetyError) as exc:
                    raise WebFetchError(
                        "Could not verify this website's robots.txt policy; page access was skipped."
                    ) from exc
                if status in {401, 403}:
                    raise WebFetchError(
                        "The website's robots.txt denies access to automated readers."
                    )
                if status in {404, 410}:
                    body = b""
                elif not 200 <= status < 300:
                    raise WebFetchError(
                        "Could not verify this website's robots.txt policy; page access was skipped."
                    )
                rules = RobotFileParser()
                rules.set_url(robots_url)
                rules.parse(body.decode("utf-8", errors="replace").splitlines())
                self._robots[origin] = rules
        if not self._robots[origin].can_fetch(USER_AGENT, url):
            raise WebFetchError("This website disallows retrieval in its robots.txt.")

    async def _request(
        self, url: str, max_bytes: int
    ) -> tuple[str, dict[str, str], bytes]:
        final_url, headers, body, status = await self._request_status(url, max_bytes)
        if not 200 <= status < 300:
            raise WebFetchError(f"The website returned HTTP {status}.")
        return final_url, headers, body

    async def _request_status(
        self, value: str, max_bytes: int, *, check_robots: bool = True
    ) -> tuple[str, dict[str, str], bytes, int]:
        url = normalize_public_url(value)
        for redirect_count in range(_MAX_REDIRECTS + 1):
            parsed = urlsplit(url)
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            addresses = await resolve_public_addresses(parsed.hostname or "", port)
            for attempt in range(3):
                try:
                    status, headers, body = await self._request_once(
                        url, addresses, max_bytes
                    )
                    break
                except httpx.TimeoutException as exc:
                    if attempt < 2:
                        await asyncio.sleep(0.2 * (2**attempt))
                        continue
                    raise WebFetchError("The webpage request timed out.") from exc
                except httpx.HTTPError as exc:
                    if attempt < 2:
                        await asyncio.sleep(0.2 * (2**attempt))
                        continue
                    raise WebFetchError("The webpage connection failed.") from exc
            if status not in _REDIRECT_STATUSES:
                return url, headers, body, status
            location = headers.get("location")
            if not location:
                raise WebFetchError("The website returned a redirect without a destination.")
            if redirect_count == _MAX_REDIRECTS:
                raise WebFetchError("The webpage exceeded the redirect limit.")
            next_url = normalize_public_url(urljoin(url, location))
            if check_robots:
                await self._check_robots(next_url)
            url = next_url
        raise WebFetchError("The webpage exceeded the redirect limit.")

    async def _request_once(
        self, url: str, addresses: tuple[str, ...], max_bytes: int
    ) -> tuple[int, dict[str, str], bytes]:
        parsed = urlsplit(url)
        timeout = httpx.Timeout(self.timeout_seconds, connect=min(5.0, self.timeout_seconds))
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html, application/xhtml+xml, text/plain;q=0.2",
            "Accept-Encoding": "identity",
        }
        if self._client is not None:
            client = self._client
            async with client.stream(
                "GET", url, headers=headers, follow_redirects=False
            ) as response:
                return await _read_response(response, max_bytes)

        transport = PinnedTransport(parsed.hostname or "", addresses)
        async with httpx.AsyncClient(
            transport=transport,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            async with client.stream("GET", url, headers=headers) as response:
                return await _read_response(response, max_bytes)


async def _read_response(
    response: httpx.Response, max_bytes: int
) -> tuple[int, dict[str, str], bytes]:
    headers = dict(response.headers)
    if response.status_code in _REDIRECT_STATUSES or response.status_code in {401, 403, 404, 410}:
        return response.status_code, headers, b""
    content_length = headers.get("content-length")
    if content_length:
        try:
            declared_length = int(content_length)
        except ValueError as exc:
            raise WebFetchError("The webpage returned an invalid content length.") from exc
        if declared_length > max_bytes:
            raise WebFetchError("The webpage exceeds the configured response-size limit.")
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise WebFetchError("The webpage exceeds the configured response-size limit.")
    return response.status_code, headers, bytes(body)


def _decode_html(body: bytes, content_type: str) -> str:
    message = Message()
    message["content-type"] = content_type
    charset = message.get_content_charset()
    if charset:
        try:
            return body.decode(charset, errors="replace")
        except LookupError:
            pass
    return body.decode("utf-8", errors="replace")
