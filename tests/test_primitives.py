"""Tests for prefill and generate primitives."""

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mlx_kv_server.cache import CacheEntry, KVCacheStore
from mlx_kv_server.primitives import generate, prefill

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model() -> MagicMock:
    model = MagicMock()
    model.return_value = MagicMock()  # forward-pass logits
    return model


def _make_tokenizer(eos_id: int = 2) -> MagicMock:
    tok = MagicMock()
    tok.eos_token_id = eos_id
    return tok


def _fake_tokens(*ids: int) -> Iterator[tuple[Any, Any]]:
    """Yield (token_id_mock, logprobs_mock) pairs as a generate_step stub."""
    for i in ids:
        m = MagicMock()
        m.__int__ = lambda self, _i=i: _i  # int(mock) == i
        yield (m, MagicMock())


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------


class TestPrefill:
    def test_creates_new_cache_entry(self) -> None:
        store = KVCacheStore()
        model = _make_model()
        fake_cache = [MagicMock(), MagicMock()]

        with (
            patch(
                "mlx_kv_server.primitives.make_prompt_cache", return_value=fake_cache
            ),
            patch("mlx_kv_server.primitives.mx.eval"),
        ):
            handle = asyncio.run(prefill([1, 2, 3], "c1", model, store, asyncio.Lock()))

        assert handle == "c1"
        assert store.has("c1")
        entry = store.get("c1")
        assert entry is not None
        assert entry.length == 3
        assert entry.cache is fake_cache

    def test_extends_existing_cache_entry(self) -> None:
        store = KVCacheStore()
        model = _make_model()
        existing_cache = [MagicMock()]
        store.put("c1", CacheEntry(cache=existing_cache, length=5))

        with patch("mlx_kv_server.primitives.mx.eval"):
            asyncio.run(prefill([10, 20], "c1", model, store, asyncio.Lock()))

        entry = store.get("c1")
        assert entry is not None
        assert entry.length == 7  # 5 existing + 2 new
        assert entry.cache is existing_cache

    def test_does_not_make_new_cache_when_entry_exists(self) -> None:
        store = KVCacheStore()
        model = _make_model()
        store.put("c1", CacheEntry(cache=[MagicMock()], length=5))

        with (
            patch("mlx_kv_server.primitives.make_prompt_cache") as mock_make,
            patch("mlx_kv_server.primitives.mx.eval"),
        ):
            asyncio.run(prefill([10], "c1", model, store, asyncio.Lock()))

        mock_make.assert_not_called()

    def test_returns_cache_id_as_handle(self) -> None:
        store = KVCacheStore()
        model = _make_model()

        with (
            patch(
                "mlx_kv_server.primitives.make_prompt_cache",
                return_value=[MagicMock()],
            ),
            patch("mlx_kv_server.primitives.mx.eval"),
        ):
            result = asyncio.run(prefill([1], "my-cache", model, store, asyncio.Lock()))

        assert result == "my-cache"

    def test_empty_token_list(self) -> None:
        store = KVCacheStore()
        model = _make_model()

        with (
            patch(
                "mlx_kv_server.primitives.make_prompt_cache",
                return_value=[MagicMock()],
            ),
            patch("mlx_kv_server.primitives.mx.eval"),
        ):
            asyncio.run(prefill([], "c1", model, store, asyncio.Lock()))

        entry = store.get("c1")
        assert entry is not None
        assert entry.length == 0


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def _collect(self, coro: Any) -> list[int]:
        async def _run() -> list[int]:
            return [tok async for tok in coro]

        return asyncio.run(_run())

    def test_raises_if_cache_not_found(self) -> None:
        store = KVCacheStore()
        model = _make_model()
        tokenizer = _make_tokenizer()

        async def _run() -> None:
            async for _ in generate(
                [1], "missing", model, tokenizer, store, asyncio.Lock()
            ):
                pass

        with pytest.raises(ValueError, match="cache not found"):
            asyncio.run(_run())

    def test_streams_tokens_until_eos(self) -> None:
        store = KVCacheStore()
        model = _make_model()
        tokenizer = _make_tokenizer(eos_id=5)
        store.put("c1", CacheEntry(cache=[MagicMock()], length=3))

        # Tokens 10, 20 then EOS (5)
        fake = list(_fake_tokens(10, 20, 5))
        with patch("mlx_kv_server.primitives.generate_step", return_value=iter(fake)):
            result = self._collect(
                generate([1], "c1", model, tokenizer, store, asyncio.Lock())
            )

        assert result == [10, 20, 5]

    def test_stops_at_max_tokens(self) -> None:
        store = KVCacheStore()
        model = _make_model()
        tokenizer = _make_tokenizer(eos_id=999)  # EOS won't appear
        store.put("c1", CacheEntry(cache=[MagicMock()], length=0))

        def _infinite() -> Iterator[tuple[Any, Any]]:
            i = 0
            while True:
                yield from _fake_tokens(i)
                i += 1

        with patch("mlx_kv_server.primitives.generate_step", return_value=_infinite()):
            result = self._collect(
                generate(
                    [1], "c1", model, tokenizer, store, asyncio.Lock(), max_tokens=3
                )
            )

        assert len(result) == 3
