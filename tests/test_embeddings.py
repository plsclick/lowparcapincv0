import unittest

import httpx

from lowparllm.ai.embeddings.ollama_embeddings import (
    EmbeddingModelMissing,
    EmbeddingProtocolError,
    OllamaEmbeddings,
)


MODEL = "nomic-embed-text"


class EmbeddingTests(unittest.IsolatedAsyncioTestCase):
    async def test_batches_inputs_through_local_embedding_api(self) -> None:
        batches: list[list[str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": MODEL}]})
            body = __import__("json").loads(request.content)
            batches.append(body["input"])
            return httpx.Response(
                200,
                json={"embeddings": [[float(len(text)), 1.0] for text in body["input"]]},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            embeddings = OllamaEmbeddings(
                "http://local", MODEL, batch_size=2, client=client
            )
            vectors = await embeddings.embed(["one", "two", "three", "four", "five"])

        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        self.assertEqual(vectors, [[3.0, 1.0], [3.0, 1.0], [5.0, 1.0], [4.0, 1.0], [4.0, 1.0]])

    async def test_missing_model_reports_manual_command_without_downloading(self) -> None:
        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request.url.path)
            return httpx.Response(200, json={"models": [{"name": "qwen3.5:4b"}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            embeddings = OllamaEmbeddings("http://local", MODEL, client=client)
            with self.assertRaisesRegex(
                EmbeddingModelMissing, "ollama pull nomic-embed-text"
            ):
                await embeddings.embed(["text"])

        self.assertEqual(requests, ["/api/tags"])

    async def test_unqualified_model_matches_ollamas_latest_tag(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/tags":
                return httpx.Response(
                    200, json={"models": [{"name": f"{MODEL}:latest"}]}
                )
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            embeddings = OllamaEmbeddings("http://local", MODEL, client=client)
            self.assertEqual(await embeddings.embed(["text"]), [[1.0, 0.0]])

    async def test_malformed_and_mismatched_vectors_are_rejected(self) -> None:
        for payload in (
            {"embeddings": [[1.0], [2.0, 3.0]]},
            {"embeddings": [[float("nan")]]},
            {"embeddings": []},
        ):
            def handler(request: httpx.Request, response_payload=payload) -> httpx.Response:
                if request.url.path == "/api/tags":
                    return httpx.Response(200, json={"models": [{"name": MODEL}]})
                return httpx.Response(200, json=response_payload)

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                embeddings = OllamaEmbeddings("http://local", MODEL, client=client)
                with self.subTest(payload=payload), self.assertRaises(EmbeddingProtocolError):
                    await embeddings.embed(["one", "two"])

    async def test_transport_timeout_is_reported(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timeout", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            embeddings = OllamaEmbeddings("http://local", MODEL, client=client)
            with self.assertRaisesRegex(RuntimeError, "did not respond"):
                await embeddings.is_available()


if __name__ == "__main__":
    unittest.main()
