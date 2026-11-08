"""Public-address-only URL validation and DNS-pinned HTTP transport."""

import asyncio
import ipaddress
import socket
from collections.abc import AsyncIterable, Iterable
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpcore
import httpx


class URLSafetyError(ValueError):
    """A URL is malformed or resolves to a restricted network destination."""


class _NetworkStream(Protocol):
    async def __aiter__(self) -> AsyncIterable[bytes]: ...

    async def aclose(self) -> None: ...


def normalize_public_url(value: str) -> str:
    """Validate URL syntax and return a canonical HTTP(S) URL."""
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise URLSafetyError("URLs containing control characters are not allowed.")
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError as exc:
        raise URLSafetyError("The URL is malformed.") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise URLSafetyError("Only absolute HTTP and HTTPS URLs are supported.")
    if parsed.username is not None or parsed.password is not None:
        raise URLSafetyError("URLs containing user credentials are not allowed.")
    expected_port = 443 if parsed.scheme.lower() == "https" else 80
    if port not in {None, expected_port}:
        raise URLSafetyError("Only standard HTTP and HTTPS ports are allowed.")
    host = parsed.hostname.rstrip(".").lower()
    if not host or host == "localhost" or host.endswith(
        (".localhost", ".local", ".internal")
    ):
        raise URLSafetyError("Local and internal network hostnames are not allowed.")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise URLSafetyError("The URL hostname is invalid.") from exc
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit(
        (
            parsed.scheme.lower(),
            netloc,
            parsed.path or "/",
            parsed.query,
            "",
        )
    )


async def resolve_public_addresses(host: str, port: int) -> tuple[str, ...]:
    """Resolve all destination addresses and reject any non-public result."""
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        addresses = (str(literal),)
    else:
        try:
            records = await asyncio.to_thread(
                socket.getaddrinfo,
                host,
                port,
                type=socket.SOCK_STREAM,
            )
        except OSError as exc:
            raise URLSafetyError("The website hostname could not be resolved.") from exc
        addresses = tuple(dict.fromkeys(record[4][0] for record in records))
    if not addresses:
        raise URLSafetyError("The website hostname resolved to no addresses.")
    for address in addresses:
        try:
            parsed_address = ipaddress.ip_address(address)
        except ValueError as exc:
            raise URLSafetyError("The website resolved to an invalid IP address.") from exc
        if (
            isinstance(parsed_address, ipaddress.IPv6Address)
            and parsed_address.ipv4_mapped is not None
        ):
            parsed_address = parsed_address.ipv4_mapped
        if (
            not parsed_address.is_global
            or parsed_address.is_multicast
            or parsed_address.is_unspecified
        ):
            raise URLSafetyError(
                "The website resolves to a private or otherwise restricted network address."
            )
    return addresses


class _PinnedNetworkBackend:
    """Connect to verified IPs while retaining the original TLS hostname."""

    def __init__(self, hostname: str, addresses: Iterable[str]) -> None:
        self._hostname = hostname.encode("idna").decode("ascii").lower().rstrip(".")
        self._addresses = tuple(addresses)
        self._backend = httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        requested = host.encode("idna").decode("ascii").lower().rstrip(".")
        if requested != self._hostname:
            raise httpcore.ConnectError("Unexpected host in pinned connection.")
        last_error: httpcore.ConnectError | None = None
        for address in self._addresses:
            try:
                return await self._backend.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except httpcore.ConnectError as exc:
                last_error = exc
        if last_error is None:
            raise httpcore.ConnectError("No verified public address is available.")
        raise last_error

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        del path, timeout, socket_options
        raise httpcore.ConnectError("Unix-domain sockets are not allowed for web requests.")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class _CoreResponseStream(httpx.AsyncByteStream):
    def __init__(self, stream: _NetworkStream) -> None:
        self._stream = stream

    async def __aiter__(self):
        async for part in self._stream:
            yield part

    async def aclose(self) -> None:
        await self._stream.aclose()


class PinnedTransport(httpx.AsyncBaseTransport):
    """HTTPX transport whose TCP connection uses the validated DNS answers."""

    def __init__(self, hostname: str, addresses: Iterable[str]) -> None:
        self._pool = httpcore.AsyncConnectionPool(
            network_backend=_PinnedNetworkBackend(hostname, addresses),
            max_connections=1,
            max_keepalive_connections=0,
            retries=0,
            http2=False,
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        core_url = httpcore.URL(
            scheme=request.url.raw_scheme,
            host=request.url.raw_host,
            port=request.url.port,
            target=request.url.raw_path,
        )
        core_request = httpcore.Request(
            method=request.method,
            url=core_url,
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        response = await self._pool.handle_async_request(core_request)
        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            stream=_CoreResponseStream(response.stream),
            extensions=response.extensions,
            request=request,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()
