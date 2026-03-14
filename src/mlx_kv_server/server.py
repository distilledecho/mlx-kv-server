"""Unix socket server and request dispatcher.

Wire protocol: newline-delimited JSON (NDJSON).

Request:
    {"id": <int>, "method": <str>, "params": <obj>}

Response (non-streaming):
    {"id": <int>, "result": <obj>}

Streaming response (generate):
    {"id": <int>, "token": <int>}   -- one per generated token
    {"id": <int>, "done": true}     -- final frame

Error response:
    {"id": <int>, "error": <str>}
"""

import asyncio
import json
import logging
import os
from typing import Any

from .cache import KVCacheStore
from .config import Config
from .primitives import generate, prefill

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj) + "\n").encode()


async def _send(writer: asyncio.StreamWriter, obj: dict[str, Any]) -> None:
    writer.write(_encode(obj))
    await writer.drain()


async def _send_error(
    writer: asyncio.StreamWriter, req_id: int | None, message: str
) -> None:
    await _send(writer, {"id": req_id, "error": message})


# ---------------------------------------------------------------------------
# Request dispatch
# ---------------------------------------------------------------------------


async def _handle_prefill(
    writer: asyncio.StreamWriter,
    req_id: int,
    params: dict[str, Any],
    model: Any,
    cache_store: KVCacheStore,
    lock: asyncio.Lock,
) -> None:
    tokens: list[int] = params["tokens"]
    cache_id: str = params["cache_id"]
    handle = await prefill(tokens, cache_id, model, cache_store, lock)
    await _send(writer, {"id": req_id, "result": {"handle": handle}})


async def _handle_generate(
    writer: asyncio.StreamWriter,
    req_id: int,
    params: dict[str, Any],
    model: Any,
    tokenizer: Any,
    cache_store: KVCacheStore,
    lock: asyncio.Lock,
    config: Config,
) -> None:
    tokens: list[int] = params["tokens"]
    cache_id: str = params["cache_id"]
    async for token in generate(
        tokens,
        cache_id,
        model,
        tokenizer,
        cache_store,
        lock,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    ):
        await _send(writer, {"id": req_id, "token": token})
    await _send(writer, {"id": req_id, "done": True})


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------


async def _handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    model: Any,
    tokenizer: Any,
    cache_store: KVCacheStore,
    lock: asyncio.Lock,
    config: Config,
) -> None:
    peer = writer.get_extra_info("peername") or "<unix>"
    logger.info("connection accepted from %s", peer)
    try:
        while True:
            raw = await reader.readline()
            if not raw:
                break  # client closed connection

            req_id: int | None = None
            try:
                request = json.loads(raw.decode())
                req_id = int(request["id"])
                method: str = request["method"]
                params: dict[str, Any] = request.get("params", {})
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                await _send_error(writer, req_id, f"malformed request: {exc}")
                continue

            try:
                if method == "prefill":
                    await _handle_prefill(
                        writer, req_id, params, model, cache_store, lock
                    )
                elif method == "generate":
                    await _handle_generate(
                        writer,
                        req_id,
                        params,
                        model,
                        tokenizer,
                        cache_store,
                        lock,
                        config,
                    )
                else:
                    await _send_error(writer, req_id, f"unknown method: {method!r}")
            except (KeyError, TypeError) as exc:
                await _send_error(writer, req_id, f"bad params: {exc}")
            except ValueError as exc:
                await _send_error(writer, req_id, str(exc))

    except asyncio.IncompleteReadError:
        pass
    except Exception:
        logger.exception("error handling connection from %s", peer)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        logger.info("connection closed: %s", peer)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_server(
    config: Config,
    model: Any,
    tokenizer: Any,
    cache_store: KVCacheStore,
) -> None:
    """Start the Unix socket server and serve forever.

    Removes a stale socket file if one exists from a previous run.

    Args:
        config: Loaded server configuration.
        model: Loaded MLX model.
        tokenizer: Loaded tokenizer.
        cache_store: Shared KV cache store.
    """
    socket_path = config.socket_path

    # Remove stale socket from a prior run.
    if os.path.exists(socket_path):
        os.remove(socket_path)

    lock = asyncio.Lock()

    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        await _handle_connection(r, w, model, tokenizer, cache_store, lock, config)

    server = await asyncio.start_unix_server(handler, path=socket_path)

    logger.info(
        "mlx-kv-server listening on %s (model=%s)", socket_path, config.model_name
    )
    async with server:
        await server.serve_forever()
