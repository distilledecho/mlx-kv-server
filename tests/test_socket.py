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
from mlx_kv_server.server import StatusTracker, _handle_connection

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
    tracker: StatusTracker | None = None,
) -> asyncio.AbstractServer:
    """Start a short-lived test server using _handle_connection."""
    _tracker = tracker if tracker is not None else StatusTracker()

    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        await _handle_connection(
            r, w, model, tokenizer, cache_store, lock, config, _tracker
        )

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


class TestStatusEndpoint:
    def test_status_returns_correct_schema(self) -> None:
        socket_path = _sock_path()

        async def run() -> dict[str, Any]:
            model = _make_model()
            tokenizer = _make_tokenizer()
            store = KVCacheStore()
            store.put("c1", CacheEntry(cache=[MagicMock()], length=512))
            lock = asyncio.Lock()
            config = _config(socket_path)
            server = await _start_server(
                socket_path, model, tokenizer, store, lock, config
            )
            async with server:
                return await _send_recv_one(
                    socket_path,
                    {"id": 1, "method": "status", "params": {}},
                )

        try:
            resp = asyncio.run(run())
        finally:
            _cleanup(socket_path)

        assert resp["id"] == 1
        result = resp["result"]
        assert result["cache_used_tokens"] == 512
        assert result["cache_capacity_tokens"] == 8192
        assert abs(result["cache_used_fraction"] - 512 / 8192) < 1e-9
        assert result["checkpoint_present"] is False
        assert result["checkpoint_tokens"] is None
        assert result["last_operation"] is None
        assert result["last_operation_at"] is None
        assert result["model"] == "test-model"
        assert isinstance(result["uptime_seconds"], int)

    def test_status_reflects_last_operation_and_checkpoint(self) -> None:
        socket_path = _sock_path()

        async def run() -> tuple[dict[str, Any], dict[str, Any]]:
            model = _make_model()
            tokenizer = _make_tokenizer()
            store = KVCacheStore()
            store.put("c1", CacheEntry(cache=[MagicMock()], length=100))
            lock = asyncio.Lock()
            config = _config(socket_path)
            tracker = StatusTracker()
            server = await _start_server(
                socket_path, model, tokenizer, store, lock, config, tracker
            )
            async with server:
                with patch(
                    "mlx_kv_server.server.checkpoint", new_callable=AsyncMock
                ) as mp:
                    mp.return_value = 100
                    await _send_recv_one(
                        socket_path,
                        {
                            "id": 1,
                            "method": "checkpoint",
                            "params": {"cache_id": "c1"},
                        },
                    )
                status_resp = await _send_recv_one(
                    socket_path,
                    {"id": 2, "method": "status", "params": {}},
                )
            return status_resp

        try:
            resp = asyncio.run(run())
        finally:
            _cleanup(socket_path)

        result = resp["result"]
        assert result["last_operation"] == "checkpoint"
        assert result["last_operation_at"] is not None
        assert result["checkpoint_present"] is True
        assert result["checkpoint_tokens"] == 100

    def test_status_checkpoint_cleared_after_evict(self) -> None:
        socket_path = _sock_path()

        async def run() -> dict[str, Any]:
            model = _make_model()
            tokenizer = _make_tokenizer()
            store = KVCacheStore()
            store.put("c1", CacheEntry(cache=[MagicMock()], length=100))
            lock = asyncio.Lock()
            config = _config(socket_path)
            tracker = StatusTracker()
            server = await _start_server(
                socket_path, model, tokenizer, store, lock, config, tracker
            )
            async with server:
                with patch(
                    "mlx_kv_server.server.checkpoint", new_callable=AsyncMock
                ) as mp:
                    mp.return_value = 100
                    await _send_recv_one(
                        socket_path,
                        {
                            "id": 1,
                            "method": "checkpoint",
                            "params": {"cache_id": "c1"},
                        },
                    )
                # Re-add entry so evict can find it
                store.put("c1", CacheEntry(cache=[MagicMock()], length=100))
                with patch("mlx_kv_server.server.evict", new_callable=AsyncMock) as me:
                    me.return_value = True
                    await _send_recv_one(
                        socket_path,
                        {
                            "id": 2,
                            "method": "evict",
                            "params": {"cache_id": "c1"},
                        },
                    )
                return await _send_recv_one(
                    socket_path,
                    {"id": 3, "method": "status", "params": {}},
                )

        try:
            resp = asyncio.run(run())
        finally:
            _cleanup(socket_path)

        result = resp["result"]
        assert result["last_operation"] == "evict"
        assert result["checkpoint_present"] is False
        assert result["checkpoint_tokens"] is None


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
