import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import tomllib
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from lowparllm.ai.rag.rag_service import RagService
from lowparllm.ai.rag.citations import build_references
from lowparllm.ai.research.research_service import ResearchService
from lowparllm.ai.search.html_extractor import extract_article
from lowparllm.ai.search.models import SearchResult, WebPage
from lowparllm.ai.search.cache import SearchCache
from lowparllm.ai.search.provider import SearchProviderError
from lowparllm.ai.search.searxng_provider import SearXNGProvider, normalize_searxng_url
from lowparllm.ai.search.url_security import (
    URLSafetyError,
    normalize_public_url,
    resolve_public_addresses,
)
from lowparllm.ai.search.web_fetcher import WebFetchError, WebPageFetcher, _read_response
from lowparllm.ai.rag.retriever import RetrievedChunk
from lowparllm.ai.llm.ollama_client import OllamaProtocolError
from lowparllm.core.config import (
    HARD_MAX_WEB_RESPONSE_BYTES,
    AppConfig,
    RagConfig,
    WebConfig,
)


_HTML = (
    "<html><head><title>Network guide</title>"
    '<meta property="article:published_time" content="2026-04-04">'
    "</head><body><nav>Navigation noise</nav><main><h1>Network guide</h1>"
    '<div class="advert"><p>Advertisement noise</p></div>'
    "<p>Public network content describes routing and congestion control.</p>"
    "<p>Additional network details explain packet delivery.</p>"
    "</main><footer>Footer noise</footer></body></html>"
)


class FakeProvider:
    name = "fake"
    configured = True

    def __init__(self, results=(), error=None):
        self.results = tuple(results)
        self.error = error
        self.queries = []
        self.closed = False
        self.cache = self
        self.configure_calls = []

    async def search(
        self,
        query,
        *,
        max_results,
        categories=(),
        language="",
        safe_search=1,
        fresh=False,
    ):
        del categories, language, safe_search, fresh
        self.queries.append((query, max_results))
        if self.error is not None:
            raise self.error
        return self.results[:max_results]

    def configure(self, **settings):
        self.configure_calls.append(settings)

    async def stats(self, *, ttl_seconds):
        return {"entries": 0, "ttl_seconds": ttl_seconds, "path": ""}

    async def clear(self):
        return 0

    async def aclose(self):
        self.closed = True


class FakePageFetcher:
    def __init__(self, pages=None, error=None):
        self.pages = pages or {}
        self.error = error
        self.urls = []

    async def fetch(self, url, *, progress=None):
        self.urls.append(url)
        if progress is not None:
            progress("Extracting webpage content")
        if self.error is not None:
            raise self.error
        return self.pages[url]


class FakeEmbeddings:
    async def is_available(self, *, refresh=False):
        del refresh
        return True

    async def aclose(self):
        return None

    async def embed_batches(self, texts):
        for _ in range(0, len(texts), 16):
            yield [[1.0, 0.0] for _ in texts[:16]]

    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeRagService:
    async def search(self, _query):
        del _query
        return type(
            "Search",
            (),
            {
                "chunks": (
                    RetrievedChunk(
                        {
                            "document_id": "local-1",
                            "filename": "notes.md",
                            "text": "My indexed notes describe local network policy.",
                            "section_heading": "Network policy",
                        },
                        0.9,
                    ),
                )
            },
        )()


def _result(url: str, *, title="Network results", score=0.8, snippet=None):
    return SearchResult(
        title=title,
        url=url,
        snippet=snippet or f"Useful details about network routing from {title}.",
        domain=httpx.URL(url).host,
        score=score,
    )


def _page(url: str, *, content="Network routing details from the source."):
    return WebPage(
        title="Network guide",
        url=url,
        canonical_url=None,
        domain=httpx.URL(url).host,
        published_date="2026-04-04",
        content=content,
        accessed_at=datetime(2026, 4, 5, tzinfo=timezone.utc),
    )


class SearXNGProviderTests(unittest.TestCase):
    def _provider(self, client: httpx.AsyncClient, name: str = "cache.sqlite3"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return SearXNGProvider(
            client=client,
            cache=SearchCache(Path(temporary.name) / name),
        )

    def test_base_url_rejects_non_loopback_http_and_malformed_input(self):
        for value in ("", "http://example.com", "http://user:pass@127.0.0.1:8080"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_searxng_url(value)
        self.assertEqual(
            normalize_searxng_url("http://127.0.0.1:8080/"),
            "http://127.0.0.1:8080",
        )
        self.assertEqual(
            normalize_searxng_url("https://search.example.org/lowp"),
            "https://search.example.org/lowp",
        )

    def test_search_normalizes_result_metadata(self):
        async def run():
            observed = {}

            async def handle(request):
                observed["params"] = dict(request.url.params)
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "title": "Official network guide",
                                "url": "https://example.com/guide",
                                "content": "Routing details.",
                                "score": 0.95,
                                "publishedDate": "2026-04-04",
                                "favicon": "https://example.com/favicon.ico",
                            }
                        ]
                    },
                )

            client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            provider = self._provider(client)
            results = await provider.search("network routing", max_results=2)
            self.assertEqual(observed["params"]["q"], "network routing")
            self.assertEqual(observed["params"]["format"], "json")
            self.assertEqual(observed["params"]["pageno"], "2")
            self.assertEqual(results[0].domain, "example.com")
            self.assertEqual(results[0].published_date, "2026-04-04")
            self.assertEqual(
                results[0].provider_metadata["favicon"],
                "https://example.com/favicon.ico",
            )
            await client.aclose()

        asyncio.run(run())

    def test_empty_results_are_normalized(self):
        async def run():
            observed = {}

            async def handle(request):
                observed.update(request.url.params)
                return httpx.Response(200, json={"results": []})

            client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            provider = self._provider(client, "empty.sqlite3")
            self.assertEqual(await provider.search("query", max_results=5), ())
            self.assertEqual(observed["format"], "json")
            await client.aclose()

        asyncio.run(run())

    def test_pagination_normalizes_metadata_and_removes_duplicate_urls(self):
        async def run():
            requested_pages = []
            observed_params = []

            async def handle(request):
                page = request.url.params["pageno"]
                requested_pages.append(page)
                observed_params.append(dict(request.url.params))
                items = (
                    [
                        {"title": "A", "url": "https://example.com/a", "content": "a"},
                        {"title": "B", "url": "https://example.com/b", "content": "b"},
                    ]
                    if page == "1"
                    else [
                        {"title": "Duplicate B", "url": "https://example.com/b/", "content": "b"},
                        {"title": "C", "url": "https://example.com/c", "content": "c"},
                    ]
                )
                return httpx.Response(200, json={"results": items})

            client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            provider = self._provider(client, "pagination.sqlite3")
            results = await provider.search(
                "query", max_results=3, categories=("general",), language="en"
            )
            self.assertEqual(requested_pages, ["1", "2"])
            self.assertEqual(observed_params[0]["categories"], "general")
            self.assertEqual(observed_params[0]["language"], "en")
            self.assertEqual(observed_params[0]["safesearch"], "1")
            self.assertEqual([result.title for result in results], ["A", "B", "C"])
            await client.aclose()

        asyncio.run(run())

    def test_cache_hit_fresh_bypass_expiration_and_clear(self):
        async def run():
            request_count = 0

            async def handle(request):
                nonlocal request_count
                request_count += 1
                self.assertEqual(request.url.params["format"], "json")
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "title": f"Result {request_count}",
                                "url": "https://example.com/",
                                "content": "cached result",
                            }
                        ]
                    },
                )

            client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            provider = self._provider(client, "cached.sqlite3")
            initial = await provider.search("same query", max_results=1)
            cached = await provider.search("same query", max_results=1)
            self.assertEqual(request_count, 1)
            self.assertEqual(cached, initial)

            fresh = await provider.search("same query", max_results=1, fresh=True)
            self.assertEqual(request_count, 2)
            self.assertEqual(fresh[0].title, "Result 2")

            with sqlite3.connect(provider.cache.path) as connection:
                connection.execute(
                    "UPDATE search_cache SET created_at = ?",
                    ((datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),),
                )
            await provider.search("same query", max_results=1)
            self.assertEqual(request_count, 3)
            stats = await provider.cache.stats(ttl_seconds=900)
            self.assertEqual(stats["entries"], 1)
            self.assertEqual(await provider.cache.clear(), 1)
            self.assertEqual(
                (await provider.cache.stats(ttl_seconds=900))["entries"], 0
            )
            await client.aclose()

        asyncio.run(run())

    def test_connection_status_reports_unavailable_instance(self):
        async def run():
            async def unavailable(_request):
                del _request
                raise httpx.ConnectError("refused")

            client = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
            provider = self._provider(client, "offline.sqlite3")
            connected, error = await provider.check_connection()
            self.assertFalse(connected)
            self.assertIn("Could not connect", error)
            await client.aclose()

        asyncio.run(run())

    def test_provider_rejects_invalid_response(self):
        async def run():
            async def unauthorized(_):
                return httpx.Response(401, text="unauthorized")

            client = httpx.AsyncClient(transport=httpx.MockTransport(unauthorized))
            provider = self._provider(client, "unauthorized.sqlite3")
            with self.assertRaises(SearchProviderError):
                await provider.search("query", max_results=5)
            await client.aclose()

            async def invalid(_):
                return httpx.Response(200, json={"results": "not-a-list"})

            client = httpx.AsyncClient(transport=httpx.MockTransport(invalid))
            provider = self._provider(client, "invalid.sqlite3")
            with self.assertRaises(SearchProviderError):
                await provider.search("query", max_results=5)
            await client.aclose()

        asyncio.run(run())

    def test_search_provider_errors_and_timeout_are_reported(self):
        async def run():
            async def unavailable(_):
                return httpx.Response(503)

            client = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
            provider = self._provider(client, "unavailable.sqlite3")
            with patch("lowparllm.ai.search.searxng_provider.asyncio.sleep", new=AsyncMock()):
                with self.assertRaises(SearchProviderError):
                    await provider.search("query", max_results=5)
            await client.aclose()

            async def timeout(_):
                raise httpx.ReadTimeout("timeout")

            client = httpx.AsyncClient(transport=httpx.MockTransport(timeout))
            provider = self._provider(client, "timeout.sqlite3")
            with patch("lowparllm.ai.search.searxng_provider.asyncio.sleep", new=AsyncMock()):
                with self.assertRaisesRegex(SearchProviderError, "timed out"):
                    await provider.search("query", max_results=5)
            await client.aclose()

        asyncio.run(run())


class WebSecurityAndExtractionTests(unittest.TestCase):
    def test_url_validation_rejects_local_hosts_credentials_and_wrong_ports(self):
        for url in (
            "file:///etc/passwd",
            "http://localhost/",
            "http://user:password@example.com/",
            "http://example.com:443/",
            "https://example.com:80/",
        ):
            with self.subTest(url=url), self.assertRaises(URLSafetyError):
                normalize_public_url(url)

    def test_dns_validation_rejects_any_non_public_address(self):
        records = [
            (2, 1, 6, "", ("1.1.1.1", 443)),
            (2, 1, 6, "", ("127.0.0.1", 443)),
        ]
        with patch(
            "lowparllm.ai.search.url_security.socket.getaddrinfo",
            return_value=records,
        ):
            with self.assertRaises(URLSafetyError):
                asyncio.run(resolve_public_addresses("public.example", 443))

    def test_article_extraction_removes_boilerplate_and_keeps_metadata(self):
        page = extract_article(
            _HTML,
            url="https://example.com/network",
            accessed_at=datetime(2026, 4, 5, tzinfo=timezone.utc),
            max_chars=5000,
        )
        self.assertEqual(page.title, "Network guide")
        self.assertEqual(page.published_date, "2026-04-04")
        self.assertIn("routing and congestion control", page.content)
        self.assertNotIn("Navigation noise", page.content)
        self.assertNotIn("Footer noise", page.content)
        self.assertNotIn("Advertisement noise", page.content)

    def test_fetcher_handles_html_robots_and_unsupported_content_type(self):
        async def run():
            async def handler(request):
                if request.url.path == "/robots.txt":
                    return httpx.Response(404)
                if request.url.path == "/plain":
                    return httpx.Response(200, headers={"content-type": "text/plain"}, text="text")
                return httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                    text=_HTML,
                )

            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            fetcher = WebPageFetcher(client=client)
            with patch(
                "lowparllm.ai.search.web_fetcher.resolve_public_addresses",
                new=AsyncMock(return_value=("93.184.216.34",)),
            ):
                page = await fetcher.fetch("https://example.com/")
                self.assertIn("routing and congestion control", page.content)
                with self.assertRaisesRegex(WebFetchError, "unsupported content type"):
                    await fetcher.fetch("https://example.com/plain")
            await client.aclose()

        asyncio.run(run())

    def test_fetcher_revalidates_redirect_destination_and_bounds_body(self):
        async def run():
            fetcher = WebPageFetcher()
            with (
                patch(
                    "lowparllm.ai.search.web_fetcher.resolve_public_addresses",
                    new=AsyncMock(
                        side_effect=[
                            ("93.184.216.34",),
                            URLSafetyError("private destination"),
                        ]
                    ),
                ),
                patch.object(
                    fetcher,
                    "_request_once",
                    new=AsyncMock(return_value=(302, {"location": "http://127.0.0.1/"}, b"")),
                ) as request,
            ):
                with self.assertRaises(WebFetchError):
                    await fetcher._request_status("https://example.com/", 1024)
                request.assert_awaited_once()

            response = httpx.Response(
                200,
                headers={"content-length": "500"},
                content=b"small",
            )
            with self.assertRaisesRegex(WebFetchError, "size limit"):
                await _read_response(response, 100)

        asyncio.run(run())

    def test_public_redirect_checks_each_destination_robots_policy(self):
        async def run():
            requests = []

            async def handler(request):
                requests.append((request.url.host, request.url.path))
                if request.url.path == "/robots.txt":
                    if request.url.host == "example.com":
                        return httpx.Response(404)
                    return httpx.Response(
                        200,
                        text="User-agent: *\nDisallow: /blocked\n",
                        headers={"content-type": "text/plain"},
                    )
                if request.url.path == "/start":
                    return httpx.Response(
                        302,
                        headers={"location": "https://other.example/blocked"},
                    )
                return httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                    text=_HTML,
                )

            client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                follow_redirects=True,
            )
            fetcher = WebPageFetcher(client=client)
            with patch(
                "lowparllm.ai.search.web_fetcher.resolve_public_addresses",
                new=AsyncMock(return_value=("93.184.216.34",)),
            ):
                with self.assertRaisesRegex(WebFetchError, "robots.txt"):
                    await fetcher.fetch("https://example.com/start")
            self.assertIn(("other.example", "/robots.txt"), requests)
            self.assertNotIn(("other.example", "/blocked"), requests)
            await client.aclose()

        asyncio.run(run())

    def test_fetcher_timeout_and_cancellation(self):
        async def run():
            fetcher = WebPageFetcher()
            with (
                patch(
                    "lowparllm.ai.search.web_fetcher.resolve_public_addresses",
                    new=AsyncMock(return_value=("93.184.216.34",)),
                ),
                patch(
                    "lowparllm.ai.search.web_fetcher.asyncio.sleep",
                    new=AsyncMock(),
                ),
                patch.object(
                    fetcher,
                    "_request_once",
                    new=AsyncMock(side_effect=httpx.ReadTimeout("timeout")),
                ) as request,
            ):
                with self.assertRaisesRegex(WebFetchError, "timed out"):
                    await fetcher._request_status("https://example.com/", 1024)
                self.assertEqual(request.await_count, 3)

        asyncio.run(run())


class ResearchServiceTests(unittest.TestCase):
    def test_ollama_query_planning_protocol_error_uses_fallback_search(self):
        async def run():
            result = _result("https://example.com/llm")
            provider = FakeProvider((result,))
            fetcher = FakePageFetcher(
                {result.url: _page(result.url, content="An LLM is a language model.")}
            )

            async def fail_to_plan(_question, _maximum):
                del _question, _maximum
                raise OllamaProtocolError(
                    "The local model did not return a usable research-query plan."
                )

            service = ResearchService(
                AppConfig(web=WebConfig(max_queries=1)),
                FakeRagService(),
                provider=provider,
                page_fetcher=fetcher,
                query_planner=fail_to_plan,
            )
            with patch(
                "lowparllm.ai.research.research_service.resolve_public_addresses",
                new=AsyncMock(return_value=("93.184.216.34",)),
            ):
                bundle = await service.research_web("what is a llm")

            self.assertTrue(bundle.has_evidence)
            self.assertEqual(provider.queries[0][0], "what is a llm")
            self.assertIn(
                "Local query planning was unavailable; using a focused fallback plan.",
                bundle.errors,
            )
            await service.close()

        asyncio.run(run())

    def test_status_uses_searxng_and_settings_are_persisted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.toml"
            path.write_text(
                '[model]\nname = "qwen3.5:4b"\n\n[web]\nenabled = true\n',
                encoding="utf-8",
            )
            config = AppConfig(web=WebConfig())
            service = ResearchService(
                config,
                FakeRagService(),
                provider=FakeProvider(),
                page_fetcher=FakePageFetcher(),
            )
            with (
                patch("lowparllm.core.config.config_dir", return_value=root),
            ):
                status = service.status()
                self.assertEqual(status["provider"], "searxng")
                self.assertNotIn("api_key_configured", status)
                service.set_setting("automatic_research", True)
            import tomllib

            saved = tomllib.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(saved["web"]["automatic_research"])
            self.assertEqual(saved["model"]["name"], "qwen3.5:4b")
            asyncio.run(service.close())

    def test_numeric_web_settings_persist_and_take_effect_immediately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.toml"
            path.write_text(
                '[model]\nname = "qwen3.5:4b"\n\n[web]\n'
                'max_pages = 5 # keep comment\n',
                encoding="utf-8",
            )
            config = AppConfig(web=WebConfig())
            fetcher = WebPageFetcher()
            service = ResearchService(
                config,
                FakeRagService(),
                provider=FakeProvider(),
                page_fetcher=fetcher,
            )
            with patch("lowparllm.core.config.config_dir", return_value=root):
                service.set_limit_setting("max_pages", "3")
                service.set_limit_setting("max_page_bytes", "1048576")
                service.set_limit_setting("max_page_chars", "8000")
                service.set_limit_setting("timeout_seconds", "12.5")
                saved = tomllib.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(service.status()["max_pages"], 3)
            self.assertEqual(service.status()["max_page_bytes"], 1048576)
            self.assertEqual(service.status()["max_page_chars"], 8000)
            self.assertEqual(service.status()["timeout_seconds"], 12.5)
            self.assertEqual(fetcher.max_page_bytes, 1048576)
            self.assertEqual(fetcher.max_page_chars, 8000)
            self.assertEqual(fetcher.timeout_seconds, 12.5)
            self.assertEqual(saved["web"]["max_pages"], 3)
            self.assertEqual(saved["web"]["max_page_bytes"], 1048576)
            self.assertEqual(saved["web"]["max_page_chars"], 8000)
            self.assertEqual(saved["web"]["timeout_seconds"], 12.5)
            self.assertIn("# keep comment", path.read_text(encoding="utf-8"))
            asyncio.run(service.close())

    def test_unlimited_mode_ignores_task_quotas_but_keeps_hard_web_safety_bounds(self):
        async def run():
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config_path = root / "config.toml"
                config_path.write_text(
                    '[web]\nmax_pages = 1\nmax_results = 1\n',
                    encoding="utf-8",
                )
                results = tuple(
                    _result(f"https://example.com/{index}", title=f"Page {index}")
                    for index in range(3)
                )
                provider = FakeProvider(results)
                pages = {
                    result.url: _page(result.url, content=f"Evidence for {result.title}")
                    for result in results
                }
                fetcher = FakePageFetcher(pages)
                planned_maxima = []

                async def planner(_question, maximum):
                    del _question
                    planned_maxima.append(maximum)
                    return ("first focused query", "second focused query")

                service = ResearchService(
                    AppConfig(
                        context_size=4096,
                        web=WebConfig(
                            max_queries=1,
                            max_results=1,
                            max_pages=1,
                            max_page_bytes=65536,
                            max_page_chars=1000,
                            max_context_chars=2000,
                            unlimited=False,
                        ),
                    ),
                    FakeRagService(),
                    provider=provider,
                    page_fetcher=fetcher,
                    query_planner=planner,
                )
                with (
                    patch("lowparllm.core.config.config_dir", return_value=root),
                    patch(
                        "lowparllm.ai.research.research_service.resolve_public_addresses",
                        new=AsyncMock(return_value=("93.184.216.34",)),
                    ),
                ):
                    service.set_setting("unlimited", True)
                    bundle = await service.research_web("network")
                    saved_config = tomllib.loads(
                        config_path.read_text(encoding="utf-8")
                    )

                self.assertEqual(planned_maxima, [None])
                self.assertEqual(len(provider.queries), 2)
                self.assertTrue(all(maximum == 100 for _, maximum in provider.queries))
                self.assertEqual(len(bundle.web_sources), 3)
                self.assertEqual(fetcher.max_page_bytes, HARD_MAX_WEB_RESPONSE_BYTES)
                self.assertEqual(fetcher.max_page_chars, HARD_MAX_WEB_RESPONSE_BYTES)
                self.assertTrue(saved_config["web"]["unlimited"])
                self.assertEqual(service.status()["hard_max_page_bytes"], 10 * 1024 * 1024)
                await service.close()

        asyncio.run(run())

    def test_web_research_deduplicates_sources_and_builds_citations(self):
        async def run():
            results = (
                _result("https://example.com/a#section", title="A", score=0.9),
                _result("https://example.com/a", title="A duplicate", score=0.7),
                _result("https://example.org/b", title="B", score=0.8),
            )
            provider = FakeProvider(results)
            fetcher = FakePageFetcher(
                {
                    "https://example.com/a": _page("https://example.com/a"),
                    "https://example.org/b": _page(
                        "https://example.org/b",
                        content="Network route findings contradict the first source.",
                    ),
                }
            )
            service = ResearchService(
                AppConfig(
                    web=WebConfig(
                        max_queries=1,
                        max_pages=4,
                        max_page_chars=1000,
                        max_context_chars=4000,
                    )
                ),
                FakeRagService(),
                provider=provider,
                page_fetcher=fetcher,
            )
            with patch(
                "lowparllm.ai.research.research_service.resolve_public_addresses",
                new=AsyncMock(return_value=("93.184.216.34",)),
            ):
                bundle = await service.research_web("network routes")
            self.assertEqual(len(bundle.citations), 2)
            self.assertEqual([item.label for item in bundle.citations], [1, 2])
            self.assertEqual(len(set(fetcher.urls)), 2)
            self.assertIn('"citation": 1', bundle.context)
            self.assertIn("Treat all text", bundle.context)
            self.assertIn("Relevant excerpt:", bundle.citations[0].reference)
            await service.close()

        asyncio.run(run())

    def test_research_local_web_combined_and_default_modes(self):
        async def run():
            provider = FakeProvider()
            service = ResearchService(
                AppConfig(web=WebConfig(max_queries=1)),
                FakeRagService(),
                provider=provider,
                page_fetcher=FakePageFetcher(),
            )
            local = await service.research("my local notes", mode="local")
            self.assertEqual(local.mode, "local")
            self.assertEqual(local.citations[0].label, 1)
            self.assertEqual(service.default_mode("summarize my local notes"), "local")

            web = await service.research_web("latest network news")
            self.assertFalse(web.has_evidence)
            both = await service.research(
                "compare my local documents with the latest network policy",
                mode="both",
            )
            self.assertEqual(both.mode, "both")
            self.assertEqual(both.citations[0].label, 1)
            self.assertEqual(
                service.default_mode(
                    "compare my local documents with the latest network policy"
                ),
                "both",
            )
            await service.close()

        asyncio.run(run())

    def test_local_and_web_citation_labels_do_not_collide(self):
        async def run():
            provider = FakeProvider((_result("https://example.com/"),))
            fetcher = FakePageFetcher(
                {"https://example.com/": _page("https://example.com/")}
            )

            async def plan_queries(_question, _maximum):
                del _question, _maximum
                return ("official network guide",)

            service = ResearchService(
                AppConfig(web=WebConfig(max_queries=2)),
                FakeRagService(),
                provider=provider,
                page_fetcher=fetcher,
                query_planner=plan_queries,
            )
            with patch(
                "lowparllm.ai.research.research_service.resolve_public_addresses",
                new=AsyncMock(return_value=("93.184.216.34",)),
            ):
                bundle = await service.research("compare my notes with current policy", "both")
            self.assertEqual([citation.label for citation in bundle.citations], [1, 2])
            self.assertEqual(bundle.web_sources[0].source_id, "2")
            self.assertEqual(provider.queries[0][0], "official network guide")
            await service.close()

        asyncio.run(run())

    def test_web_snippet_fallback_and_research_cancellation(self):
        async def run():
            provider = FakeProvider(
                (_result("https://example.com/", snippet="Network source snippet."),)
            )
            fetcher = FakePageFetcher(error=WebFetchError("page blocked"))
            service = ResearchService(
                AppConfig(web=WebConfig(max_queries=1)),
                FakeRagService(),
                provider=provider,
                page_fetcher=fetcher,
            )
            with patch(
                "lowparllm.ai.research.research_service.resolve_public_addresses",
                new=AsyncMock(return_value=("93.184.216.34",)),
            ):
                bundle = await service.research_web("network")
            self.assertEqual(bundle.citations[0].evidence_type, "search snippet")
            self.assertIn("Search-result snippet only", bundle.citations[0].reference)
            await service.close()

            class BlockingProvider(FakeProvider):
                async def search(self, query, *, max_results):
                    del query, max_results
                    await asyncio.Event().wait()

            blocking = ResearchService(
                AppConfig(web=WebConfig(max_queries=1)),
                FakeRagService(),
                provider=BlockingProvider(),
                page_fetcher=FakePageFetcher(),
            )
            task = asyncio.create_task(blocking.research_web("network"))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await blocking.close()

        asyncio.run(run())

    def test_explicit_web_save_reuses_rag_indexing_and_keeps_provenance(self):
        async def run():
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = AppConfig(
                    rag=RagConfig(
                        embedding_model="fake-embed",
                        chunk_size=100,
                        chunk_overlap=10,
                        qdrant_path=root / "qdrant",
                        sqlite_path=root / "knowledge.db",
                    ),
                    web=WebConfig(max_page_chars=1000),
                )
                rag = RagService(config, embeddings=FakeEmbeddings())
                page_fetcher = FakePageFetcher(
                    {
                        "https://example.com/article": WebPage(
                            title="Network guide",
                            url="https://example.com/final",
                            canonical_url="https://example.com/canonical",
                            domain="example.com",
                            published_date="2026-04-04",
                            content="Network routing details from the source.",
                            accessed_at=datetime(2026, 4, 5, tzinfo=timezone.utc),
                        )
                    }
                )
                service = ResearchService(
                    config,
                    rag,
                    provider=FakeProvider(),
                    page_fetcher=page_fetcher,
                )
                await service.open("https://example.com/article")
                result = await service.save_source("1")
                self.assertEqual(result.status, "indexed")
                sources, total = await rag.list_sources()
                self.assertEqual(total, 1)
                self.assertEqual(
                    sources[0].metadata["web_source"]["url"],
                    "https://example.com/article",
                )
                self.assertEqual(
                    sources[0].metadata["web_source"]["final_url"],
                    "https://example.com/final",
                )
                indexed = await rag.search("Network routing")
                _, citations = build_references(list(indexed.chunks))
                self.assertIn("https://example.com/article", citations[0].reference)
                self.assertIn("Relevant excerpt:", citations[0].reference)
                source_path = Path(sources[0].original_path)
                self.assertTrue(source_path.is_relative_to(root / "web-sources"))
                saved_content = source_path.read_text()
                self.assertIn("Source URL: https://example.com/article", saved_content)
                self.assertIn("Retrieved URL: https://example.com/final", saved_content)
                self.assertIn(
                    "Canonical URL: https://example.com/canonical", saved_content
                )
                await service.close()
                await rag.close()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
