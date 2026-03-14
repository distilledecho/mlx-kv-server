"""Tests for Unix socket server: request dispatch and error handling."""

import asyncio
import json
import os
import tempfile
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from mlx_kv_server.cache import CacheEntry, KVCacheStore
from mlx_kv_server.config import Config
from mlx_kv_server.server import _handle_connection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sock_path() -> str:
    """Return a short socket path in /tmp (AF_UNIX limit: 104 bytes on macOS)."""
    return tempfile.mktemp(prefix="mlxt", suffix=".sock", dir="/tmp")  # noqa: S306


def _config(socket_path: str) -> Config:
    return Config(socket_path=socket_path, model_name="test-model")


def _make_model() -> MagicMock:
    m = MagicMock()
    m.make_cache.return_value = [MagicMock()]
    return m


def _make_tokenizer(eos_id: int = 2) -> MagicMock:
    t = MagicMock()
    t.eos_token_id = eos_id
    return t


async def _start_server(
    socket_path: str,
    model: Any,
    tokenizer: Any,
    cache_store: KVCacheStore,
    lock: asyncio.Lock,
    config: Config,
) -> asyncio.AbstractServer:
    """Start a short-lived test server using _handle_connection."""

    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        await _handle_connection(r, w, model, tokenizer, cache_store, lock, config)

    return await asyncio.start_unix_server(handler, path=socket_path)


async def _send_recv_one(socket_path: str, request: dict[str, Any]) -> dict[str, Any]:
    """Send one request, read one response line."""
    reader, writer = await asyncio.open_unix_connection(path=socket_path)
    try:
        writer.write((json.dumps(request) + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        return json.loads(line.decode())  # type: ignore[no-any-return]
    finally:
        writer.close()
        await writer.wait_closed()


async def _send_recv_all(
    socket_path: str, request: dict[str, Any]
) -> list[dict[str, Any]]:
    """Send one request, read all response lines until done/error frame."""
    reader, writer = await asyncio.open_unix_connection(path=socket_path)
    try:
        writer.write((json.dumps(request) + "\n").encode())
        await writer.drain()
        responses: list[dict[str, Any]] = []
        while True:
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            except TimeoutError:
                break
            if not line:
                break
            responses.append(json.loads(line.decode()))
            if responses[-1].get("done") or responses[-1].get("error"):
                break
        return responses
    finally:
        writer.close()
        await writer.wait_closed()


def _cleanup(socket_path: str) -> None:
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMalformedRequests:
    def test_malformed_json_returns_error(self) -> None:
        socket_path = _sock_path()

        async def run() -> dict[str, Any]:
            model = _make_model()
            tokenizer = _make_tokenizer()
            store = KVCacheStore()
            lock = asyncio.Lock()
            config = _config(socket_path)
            server = await _start_server(
                socket_path, model, tokenizer, store, lock, config
            )
            async with server:
                reader, writer = await asyncio.open_unix_connection(path=socket_path)
                try:
                    writer.write(b"not json at all\n")
                    await writer.drain()
                    line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                    return json.loads(line.decode())
                finally:
                    writer.close()
                    await writer.wait_closed()

        try:
            resp = asyncio.run(run())
        finally:
            _cleanup(socket_path)

        assert "error" in resp
        assert resp["id"] is None

    def test_unknown_method_returns_error(self) -> None:
        socket_path = _sock_path()

        async def run() -> dict[str, Any]:
            model = _make_model()
            tokenizer = _make_tokenizer()
            store = KVCacheStore()
            lock = asyncio.Lock()
            config = _config(socket_path)
            server = await _start_server(
                socket_path, model, tokenizer, store, lock, config
            )
            async with server:
                return await _send_recv_one(
                    socket_path,
                    {"id": 1, "method": "nonexistent", "params": {}},
                )

        try:
            resp = asyncio.run(run())
        finally:
            _cleanup(socket_path)

        assert resp["id"] == 1
        assert "error" in resp
        assert "nonexistent" in resp["error"]

    def test_missing_method_field_returns_error(self) -> None:
        socket_path = _sock_path()

        async def run() -> dict[str, Any]:
            model = _make_model()
            tokenizer = _make_tokenizer()
            store = KVCacheStore()
            lock = asyncio.Lock()
            config = _config(socket_path)
            server = await _start_server(
                socket_path, model, tokenizer, store, lock, config
            )
            async with server:
                # Missing "method" key
                return await _send_recv_one(
                    socket_path,
                    {"id": 2, "params": {}},
                )

        try:
            resp = asyncio.run(run())
        finally:
            _cleanup(socket_path)

        assert resp["id"] == 2
        assert "error" in resp


class TestPrefillRequests:
    def test_prefill_returns_handle(self) -> None:
        socket_path = _sock_path()

        async def run() -> dict[str, Any]:
            model = _make_model()
            tokenizer = _make_tokenizer()
            store = KVCacheStore()
            lock = asyncio.Lock()
            config = _config(socket_path)
            server = await _start_server(
                socket_path, model, tokenizer, store, lock, config
            )
            async with server:
                with patch(
                    "mlx_kv_server.server.prefill", new_callable=AsyncMock
                ) as mp:
                    mp.return_value = "cache-abc"
                    return await _send_recv_one(
                        socket_path,
                        {
                            "id": 1,
                            "method": "prefill",
                            "params": {
                                "tokens": [1, 2, 3],
                                "cache_id": "cache-abc",
                            },
                        },
                    )

        try:
            resp = asyncio.run(run())
        finally:
            _cleanup(socket_path)

        assert resp["id"] == 1
        assert resp.get("result", {}).get("handle") == "cache-abc"

    def test_prefill_missing_params_returns_error(self) -> None:
        socket_path = _sock_path()

        async def run() -> dict[str, Any]:
            model = _make_model()
            tokenizer = _make_tokenizer()
            store = KVCacheStore()
            lock = asyncio.Lock()
            config = _config(socket_path)
            server = await _start_server(
                socket_path, model, tokenizer, store, lock, config
            )
            async with server:
                return await _send_recv_one(
                    socket_path,
                    {"id": 3, "method": "prefill", "params": {}},
                )

        try:
            resp = asyncio.run(run())
        finally:
            _cleanup(socket_path)

        assert resp["id"] == 3
        assert "error" in resp


class TestGenerateRequests:
    def test_generate_streams_tokens_then_done(self) -> None:
        socket_path = _sock_path()

        async def fake_generate(*args: Any, **kwargs: Any) -> AsyncGenerator[int, None]:
            for tok in [10, 20, 30]:
                yield tok

        async def run() -> list[dict[str, Any]]:
            model = _make_model()
            tokenizer = _make_tokenizer()
            store = KVCacheStore()
            store.put("c1", CacheEntry(cache=[MagicMock()], length=0))
            lock = asyncio.Lock()
            config = _config(socket_path)
            server = await _start_server(
                socket_path, model, tokenizer, store, lock, config
            )
            async with server:
                with patch("mlx_kv_server.server.generate", new=fake_generate):
                    return await _send_recv_all(
                        socket_path,
                        {
                            "id": 1,
                            "method": "generate",
                            "params": {"tokens": [1], "cache_id": "c1"},
                        },
                    )

        try:
            frames = asyncio.run(run())
        finally:
            _cleanup(socket_path)

        tokens = [f["token"] for f in frames if "token" in f]
        done_frames = [f for f in frames if f.get("done")]
        assert tokens == [10, 20, 30]
        assert len(done_frames) == 1
        assert done_frames[0]["id"] == 1

    def test_generate_unknown_cache_returns_error(self) -> None:
        socket_path = _sock_path()

        async def run() -> dict[str, Any]:
            model = _make_model()
            tokenizer = _make_tokenizer()
            store = KVCacheStore()  # empty — cache_id won't be found
            lock = asyncio.Lock()
            config = _config(socket_path)
            server = await _start_server(
                socket_path, model, tokenizer, store, lock, config
            )
            async with server:
                return await _send_recv_one(
                    socket_path,
                    {
                        "id": 2,
                        "method": "generate",
                        "params": {"tokens": [1], "cache_id": "no-such"},
                    },
                )

        try:
            resp = asyncio.run(run())
        finally:
            _cleanup(socket_path)

        assert resp["id"] == 2
        assert "error" in resp


class TestServerSocketCleanup:
    def test_stale_socket_removed_on_start(self) -> None:
        """run_server removes a leftover socket file before binding."""
        from mlx_kv_server.server import run_server

        socket_path = _sock_path()
        # Create a fake stale file at the socket path
        with open(socket_path, "w") as f:
            f.write("")

        assert os.path.exists(socket_path)

        async def run() -> None:
            config = _config(socket_path)
            model = _make_model()
            tokenizer = _make_tokenizer()
            store = KVCacheStore()
            task = asyncio.create_task(run_server(config, model, tokenizer, store))
            await asyncio.sleep(0.05)  # let server bind
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        try:
            asyncio.run(run())
        finally:
            _cleanup(socket_path)
        # Reaching here means no FileExistsError / OSError was raised.
