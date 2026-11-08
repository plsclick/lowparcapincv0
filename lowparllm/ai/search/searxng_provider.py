"""Asynchronous search client for an explicitly configured SearXNG instance."""

import asyncio
from collections.abc import Mapping
import hashlib
import ipaddress
import json
import math
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from lowparllm.ai.search.cache import SearchCache
from lowparllm.ai.search.models import SearchResult
from lowparllm.ai.search.provider import SearchProviderError

_RESPONSE_LIMIT = 4 * 1024 * 1024
_MAX_PAGINATION_PAGES = 10


def normalize_searxng_url(value: str) -> str:
    """Allow loopback HTTP or explicitly configured HTTPS instances."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("SEARXNG_BASE_URL must be a non-empty URL.")
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError as exc:
        raise ValueError("SEARXNG_BASE_URL is malformed.") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "SEARXNG_BASE_URL must be an HTTP(S) origin or base path without credentials, "
            "query, or fragment."
        )
    host = parsed.hostname.rstrip(".").lower()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    is_loopback = host == "localhost" or (
        address is not None and address.is_loopback
    )
    if parsed.scheme == "http" and not is_loopback:
        raise ValueError(
            "Remote SearXNG instances must use HTTPS; HTTP is allowed only on loopback."
        )
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("SEARXNG_BASE_URL has an invalid port.")
    path = parsed.path.rstrip("/")
    netloc = parsed.netloc.lower()
    return urlunsplit((parsed.scheme, netloc, path, "", ""))


class SearXNGProvider:
    name = "searxng"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        timeout_seconds: float = 10.0,
        cache: SearchCache | None = None,
        cache_ttl_seconds: int = 900,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = normalize_searxng_url(base_url)
        self.timeout_seconds = timeout_seconds
        self.cache = cache or SearchCache()
        self.cache_ttl_seconds = cache_ttl_seconds
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=min(5.0, timeout_seconds)),
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )
        self._owns_client = client is None

    @property
    def configured(self) -> bool:
        return True

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/search"

    def configure(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        cache_ttl_seconds: int,
    ) -> None:
        normalized = normalize_searxng_url(base_url)
        if normalized != self.base_url:
            self.base_url = normalized
        if timeout_seconds != self.timeout_seconds:
            self.timeout_seconds = timeout_seconds
            self._client.timeout = httpx.Timeout(
                timeout_seconds, connect=min(5.0, timeout_seconds)
            )
        self.cache_ttl_seconds = cache_ttl_seconds

    async def check_connection(self) -> tuple[bool, str | None]:
        try:
            async with self._client.stream("GET", f"{self.base_url}/") as response:
                status_code = response.status_code
        except httpx.TimeoutException:
            return False, "Connection to SearXNG timed out."
        except httpx.HTTPError:
            return False, "Could not connect to SearXNG. Is the instance running?"
        if not 200 <= status_code < 300:
            return False, f"SearXNG returned HTTP {status_code}."
        return True, None

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        categories: tuple[str, ...] = (),
        language: str = "",
        safe_search: int = 1,
        fresh: bool = False,
    ) -> tuple[SearchResult, ...]:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            return ()
        if not 1 <= max_results <= 100:
            raise ValueError("SearXNG max_results must be between 1 and 100.")
        if safe_search not in {0, 1, 2}:
            raise ValueError("SearXNG safe_search must be 0, 1, or 2.")
        parameters: dict[str, object] = {
            "base_url": self.base_url,
            "categories": tuple(sorted(set(categories))),
            "language": language,
            "safe_search": safe_search,
            "max_results": max_results,
        }
        cache_key = hashlib.sha256(
            json.dumps(
                [normalized_query.casefold(), parameters],
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if not fresh:
            cached = await self.cache.get(
                cache_key, ttl_seconds=self.cache_ttl_seconds
            )
            if cached is not None:
                return cached[:max_results]

        collected: list[SearchResult] = []
        seen_urls: set[str] = set()
        for page in range(1, _MAX_PAGINATION_PAGES + 1):
            params: dict[str, str] = {
                "q": normalized_query,
                "format": "json",
                "pageno": str(page),
                "safesearch": str(safe_search),
            }
            if categories:
                params["categories"] = ",".join(sorted(set(categories)))
            if language:
                params["language"] = language
            payload = await self._request_json(params)
            page_results = _normalize_results(payload)
            new_results = 0
            for result in page_results:
                key = result.url.split("#", 1)[0].rstrip("/").casefold()
                if key not in seen_urls:
                    seen_urls.add(key)
                    collected.append(result)
                    new_results += 1
                    if len(collected) >= max_results:
                        break
            if len(collected) >= max_results or new_results == 0:
                break
        results = tuple(collected[:max_results])
        if self.cache_ttl_seconds > 0:
            await self.cache.put(cache_key, normalized_query, parameters, results)
        return results

    async def _request_json(self, params: dict[str, str]) -> Mapping[str, Any]:
        for attempt in range(3):
            try:
                async with self._client.stream("GET", self.endpoint, params=params) as response:
                    status = response.status_code
                    if (status == 429 or status >= 500) and attempt < 2:
                        await response.aclose()
                        await asyncio.sleep(0.2 * (2**attempt))
                        continue
                    if not 200 <= status < 300:
                        raise SearchProviderError(
                            f"SearXNG returned HTTP {status}. Confirm its JSON format is enabled."
                        )
                    length = response.headers.get("content-length")
                    if length is not None:
                        try:
                            if int(length) > _RESPONSE_LIMIT:
                                raise SearchProviderError(
                                    "SearXNG response exceeded the 4 MiB safety limit."
                                )
                        except ValueError as exc:
                            raise SearchProviderError(
                                "SearXNG returned an invalid response length."
                            ) from exc
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > _RESPONSE_LIMIT:
                            raise SearchProviderError(
                                "SearXNG response exceeded the 4 MiB safety limit."
                            )
                    try:
                        payload: Any = json.loads(body)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise SearchProviderError(
                            "SearXNG did not return valid JSON. Check that format=json is enabled."
                        ) from exc
                    if not isinstance(payload, Mapping) or not isinstance(
                        payload.get("results"), list
                    ):
                        raise SearchProviderError(
                            "SearXNG JSON response must contain a results array."
                        )
                    return payload
            except httpx.TimeoutException as exc:
                if attempt < 2:
                    await asyncio.sleep(0.2 * (2**attempt))
                    continue
                raise SearchProviderError("SearXNG search timed out.") from exc
            except httpx.HTTPError as exc:
                if attempt < 2:
                    await asyncio.sleep(0.2 * (2**attempt))
                    continue
                raise SearchProviderError(
                    "Could not connect to SearXNG. Check the configured instance."
                ) from exc
        raise SearchProviderError("SearXNG search failed after retries.")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _normalize_results(payload: Mapping[str, Any]) -> tuple[SearchResult, ...]:
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise SearchProviderError("SearXNG JSON response must contain a results array.")
    results: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, Mapping):
            continue
        title, url = item.get("title"), item.get("url")
        content = item.get("content", "")
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(url, str) or not url.strip():
            continue
        if not isinstance(content, str):
            content = ""
        try:
            parsed_url = urlsplit(url.strip())
        except ValueError:
            continue
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
            continue
        positions = item.get("positions")
        position = None
        if isinstance(positions, list):
            position = next(
                (
                    value
                    for value in positions
                    if isinstance(value, int) and not isinstance(value, bool) and value > 0
                ),
                None,
            )
        published = item.get("publishedDate")
        engine = item.get("engine")
        category = item.get("category")
        score = item.get("score")
        results.append(
            SearchResult(
                title=title.strip(),
                url=url.strip(),
                snippet=content.strip(),
                domain=parsed_url.hostname,
                published_date=published if isinstance(published, str) else None,
                score=(
                    float(score)
                    if isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and math.isfinite(score)
                    else None
                ),
                engine=engine if isinstance(engine, str) else None,
                category=category if isinstance(category, str) else None,
                position=position,
                provider_metadata={
                    key: value
                    for key, value in item.items()
                    if key not in {
                        "title", "url", "content", "publishedDate", "engine",
                        "category", "positions", "score",
                    }
                    and isinstance(value, (str, bool, int, float))
                },
            )
        )
    return tuple(results)
