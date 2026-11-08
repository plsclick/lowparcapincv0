import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from lowparllm.ai.llm.generation_config import GenerationConfig
from lowparllm.ai.llm.models import ChatMessage
from lowparllm.ai.llm.ollama_client import (
    OllamaClient,
    OllamaModelMissing,
    OllamaProtocolError,
    OllamaRequestError,
    OllamaUnavailable,
)

MODEL = "qwen3.5:4b"
MESSAGES: list[ChatMessage] = [
    {"role": "system", "content": "system"},
    {"role": "user", "content": "hello"},
]
GENERATION = GenerationConfig(0.7, 0.8, 4096, -1, "5m", 20)


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


class PausingStream(ChunkStream):
    def __init__(self):
        super().__init__([])
        self.started = asyncio.Event()

    async def __aiter__(self):
        yield b'{"message":{"content":"first"}}\n'
        self.started.set()
        await asyncio.Event().wait()


class OllamaClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_status_checks_model_details_and_loaded_state(self):
        async def handler(request):
            if request.url.path == "/api/tags":
                return httpx.Response(
                    200,
                    json={"models": [{"name": MODEL, "size": 123}]},
                )
            if request.url.path == "/api/ps":
                return httpx.Response(200, json={"models": [{"name": MODEL}]})
            if request.url.path == "/api/show":
                return httpx.Response(
                    200,
                    json={"details": {"family": "qwen", "parameter_size": "4B"}},
                )
            raise AssertionError(request.url.path)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            ollama = OllamaClient("http://local", MODEL, client=client)
            with patch("lowparllm.ai.llm.ollama_client.shutil.which", return_value="/usr/bin/ollama"):
                status = await ollama.status()
        self.assertTrue(status.installed)
        self.assertTrue(status.connected)
        self.assertTrue(status.model_available)
        self.assertTrue(status.model_loaded)
        self.assertEqual(status.model_size, 123)
        self.assertIn("4B", status.model_details)

    async def test_unreachable_endpoint_is_reported(self):
        def handler(request):
            raise httpx.ConnectError("offline", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            status = await OllamaClient("http://local", MODEL, client=client).status()
        self.assertFalse(status.connected)
        self.assertFalse(status.model_available)
        self.assertIsNone(status.model_loaded)

    async def test_missing_exact_model_is_reported_without_chat_request(self):
        requests = []

        async def handler(request):
            requests.append(request.url.path)
            return httpx.Response(200, json={"models": [{"name": "qwen3.5:4b-other"}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            ollama = OllamaClient("http://local", MODEL, client=client)
            with self.assertRaises(OllamaModelMissing):
                async for _ in ollama.stream_chat(MESSAGES, GENERATION):
                    pass
        self.assertEqual(requests, ["/api/tags"])

    async def test_streaming_ndjson_content_metrics_and_resource_cleanup(self):
        body = (
            json.dumps({"message": {"content": "Hello ", "thinking": "hidden"}})
            + "\n"
            + json.dumps({"message": {"content": "世界"}})
            + "\n"
            + json.dumps(
                {
                    "done": True,
                    "eval_count": 8,
                    "eval_duration": 2_000_000_000,
                    "total_duration": 3_000_000_000,
                }
            )
            + "\n"
        ).encode()
        stream = ChunkStream([body[:35], body[35:]])
        request_body = {}

        async def handler(request):
            nonlocal request_body
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": MODEL}]})
            request_body = json.loads(request.content)
            return httpx.Response(200, stream=stream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            ollama = OllamaClient("http://local", MODEL, client=client)
            events = [event async for event in ollama.stream_chat(MESSAGES, GENERATION)]
        self.assertEqual("".join(event.content for event in events), "Hello 世界")
        self.assertEqual([event.content for event in events[:-1]], ["Hello ", "世界"])
        self.assertEqual(events[-1].metrics.eval_count, 8)
        self.assertEqual(events[-1].metrics.tokens_per_second, 4)
        self.assertTrue(stream.closed)
        self.assertEqual(request_body["model"], MODEL)
        self.assertEqual(request_body["options"]["num_ctx"], 4096)
        self.assertEqual(request_body["options"]["num_predict"], -1)
        self.assertTrue(request_body["stream"])
        self.assertFalse(request_body["think"])
        self.assertNotIn("thinking", "".join(event.content for event in events))

    async def test_malformed_stream_is_reported_and_closed(self):
        stream = ChunkStream([b"not-json\n"])

        async def handler(request):
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": MODEL}]})
            return httpx.Response(200, stream=stream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            ollama = OllamaClient("http://local", MODEL, client=client)
            with self.assertRaises(OllamaProtocolError):
                async for _ in ollama.stream_chat(MESSAGES, GENERATION):
                    pass
        self.assertTrue(stream.closed)

    async def test_cancellation_closes_http_stream(self):
        stream = PausingStream()

        async def handler(request):
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": MODEL}]})
            return httpx.Response(200, stream=stream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            ollama = OllamaClient("http://local", MODEL, client=client)
            events = ollama.stream_chat(MESSAGES, GENERATION)
            first = await anext(events)
            self.assertEqual(first.content, "first")
            pending = asyncio.create_task(anext(events))
            await stream.started.wait()
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            await events.aclose()

        self.assertTrue(stream.closed)

    async def test_api_error_model_missing_and_other_generation_error(self):
        for body, error in (
            ({"error": "model 'missing' not found"}, OllamaModelMissing),
            ({"error": "runner failed to load model"}, OllamaRequestError),
        ):
            async def handler(request, error_body=body):
                if request.url.path == "/api/tags":
                    return httpx.Response(200, json={"models": [{"name": MODEL}]})
                return httpx.Response(500, json=error_body)

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                ollama = OllamaClient("http://local", MODEL, client=client)
                with self.assertRaises(error):
                    async for _ in ollama.stream_chat(MESSAGES, GENERATION):
                        pass

    async def test_timeout_and_interrupted_stream_are_translated(self):
        async def timeout_handler(request):
            if request.url.path == "/api/tags":
                raise httpx.ReadTimeout("timeout", request=request)
            raise AssertionError("chat must not run when availability check times out")

        async with httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler)) as client:
            ollama = OllamaClient("http://local", MODEL, client=client)
            with self.assertRaises(OllamaUnavailable):
                async for _ in ollama.stream_chat(MESSAGES, GENERATION):
                    pass

        class InterruptedStream(ChunkStream):
            async def __aiter__(self):
                yield b'{"message":{"content":"partial"}}\n'
                raise httpx.ReadError(
                    "disconnected",
                    request=httpx.Request("POST", "http://local/api/chat"),
                )

        stream = InterruptedStream([])

        async def interrupted_handler(request):
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": MODEL}]})
            return httpx.Response(200, stream=stream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(interrupted_handler)) as client:
            ollama = OllamaClient("http://local", MODEL, client=client)
            with self.assertRaises(OllamaUnavailable):
                async for _ in ollama.stream_chat(MESSAGES, GENERATION):
                    pass
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
