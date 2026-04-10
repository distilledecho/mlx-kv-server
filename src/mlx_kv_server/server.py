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
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .cache import KVCacheStore
from .config import Config
from .primitives import checkpoint, evict, generate, prefill, rollback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status tracking
# ---------------------------------------------------------------------------


@dataclass
class StatusTracker:
    """Lightweight, read-only-observable record of server activity.

    checkpoint_present / checkpoint_tokens track a single global position,
    not per-cache-id state.  This is acceptable because mlx-kv-server is
    single-daemon: only one cache_id is active at a time in normal operation.
    If multiple cache_ids are ever in use simultaneously the checkpoint fields
    will reflect only the most recent checkpoint call, regardless of which
    cache_id it came from.
    """

    start_time: float = field(default_factory=time.monotonic)
    last_operation: str | None = None
    last_operation_at: datetime | None = None
    last_checkpoint_position: int | None = None

    def record(self, operation: str) -> None:
        self.last_operation = operation
        self.last_operation_at = datetime.now(UTC)

    def record_checkpoint(self, position: int) -> None:
        self.last_checkpoint_position = position
        self.record("checkpoint")

    def clear_checkpoint(self) -> None:
        """Clear the stored checkpoint position.

        Called from two distinct code paths for different semantic reasons:
        - rollback: the cache position has changed, making the prior checkpoint
          position a stale reference that may no longer be reachable.
        - evict: the cache entry is gone entirely, so the checkpoint is moot.
        """
        self.last_checkpoint_position = None


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
    tracker: StatusTracker,
) -> None:
    tokens: list[int] = params["tokens"]
    cache_id: str = params["cache_id"]
    handle = await prefill(tokens, cache_id, model, cache_store, lock)
    tracker.record("prefill")
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
    tracker: StatusTracker,
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
    tracker.record("generate")
    await _send(writer, {"id": req_id, "done": True})


async def _handle_checkpoint(
    writer: asyncio.StreamWriter,
    req_id: int,
    params: dict[str, Any],
    cache_store: KVCacheStore,
    tracker: StatusTracker,
) -> None:
    cache_id: str = params["cache_id"]
    position = await checkpoint(cache_id, cache_store)
    tracker.record_checkpoint(position)
    await _send(writer, {"id": req_id, "result": {"position": position}})


async def _handle_rollback(
    writer: asyncio.StreamWriter,
    req_id: int,
    params: dict[str, Any],
    cache_store: KVCacheStore,
    tracker: StatusTracker,
) -> None:
    cache_id: str = params["cache_id"]
    position: int = int(params["position"])
    restored = await rollback(cache_id, position, cache_store)
    tracker.clear_checkpoint()
    tracker.record("rollback")
    await _send(writer, {"id": req_id, "result": {"position": restored}})


async def _handle_evict(
    writer: asyncio.StreamWriter,
    req_id: int,
    params: dict[str, Any],
    cache_store: KVCacheStore,
    tracker: StatusTracker,
) -> None:
    cache_id: str = params["cache_id"]
    await evict(cache_id, cache_store)
    tracker.clear_checkpoint()
    tracker.record("evict")
    await _send(writer, {"id": req_id, "result": {"freed": cache_id}})


async def _handle_status(
    writer: asyncio.StreamWriter,
    req_id: int,
    cache_store: KVCacheStore,
    config: Config,
    tracker: StatusTracker,
) -> None:
    used = cache_store.total_tokens()
    capacity = config.cache_capacity_tokens
    checkpoint_pos = tracker.last_checkpoint_position
    last_op_at = tracker.last_operation_at
    await _send(
        writer,
        {
            "id": req_id,
            "result": {
                "cache_used_tokens": used,
                "cache_capacity_tokens": capacity,
                # Informational only — total_tokens() is not capped, so clamp
                # to 1.0 rather than returning a fraction > 1 when over capacity.
                "cache_used_fraction": (
                    min(used / capacity, 1.0) if capacity > 0 else 0.0
                ),
                "checkpoint_present": checkpoint_pos is not None,
                "checkpoint_tokens": checkpoint_pos,
                "last_operation": tracker.last_operation,
                "last_operation_at": (
                    last_op_at.isoformat() if last_op_at is not None else None
                ),
                "model": config.model_name,
                "uptime_seconds": int(time.monotonic() - tracker.start_time),
            },
        },
    )


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
    tracker: StatusTracker,
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
                        writer, req_id, params, model, cache_store, lock, tracker
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
                        tracker,
                    )
                elif method == "checkpoint":
                    await _handle_checkpoint(
                        writer, req_id, params, cache_store, tracker
                    )
                elif method == "rollback":
                    await _handle_rollback(writer, req_id, params, cache_store, tracker)
                elif method == "evict":
                    await _handle_evict(writer, req_id, params, cache_store, tracker)
                elif method == "status":
                    await _handle_status(writer, req_id, cache_store, config, tracker)
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
    tracker = StatusTracker()

    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        await _handle_connection(
            r, w, model, tokenizer, cache_store, lock, config, tracker
        )

    server = await asyncio.start_unix_server(handler, path=socket_path)

    logger.info(
        "mlx-kv-server listening on %s (model=%s)", socket_path, config.model_name
    )
    async with server:
        await server.serve_forever()
