"""Tests for prefill and generate primitives."""

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mlx_kv_server.cache import CacheEntry, KVCacheStore
from mlx_kv_server.primitives import checkpoint, evict, generate, prefill, rollback

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


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_returns_current_length(self) -> None:
        store = KVCacheStore()
        store.put("c1", CacheEntry(cache=[MagicMock()], length=10))
        position = asyncio.run(checkpoint("c1", store))
        assert position == 10

    def test_returns_zero_for_empty_cache(self) -> None:
        store = KVCacheStore()
        store.put("c1", CacheEntry(cache=[MagicMock()], length=0))
        position = asyncio.run(checkpoint("c1", store))
        assert position == 0

    def test_raises_if_cache_not_found(self) -> None:
        store = KVCacheStore()
        with pytest.raises(ValueError, match="cache not found"):
            asyncio.run(checkpoint("missing", store))

    def test_does_not_mutate_store(self) -> None:
        store = KVCacheStore()
        entry = CacheEntry(cache=[MagicMock()], length=5)
        store.put("c1", entry)
        asyncio.run(checkpoint("c1", store))
        assert store.get("c1") is entry
        assert store.get("c1").length == 5  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


def _make_trimmable_layer(offset: int) -> MagicMock:
    """Return a mock KVCache layer that tracks trim calls."""
    layer = MagicMock()
    layer.is_trimmable.return_value = True
    layer.trim = MagicMock(side_effect=lambda n: None)
    return layer


class TestRollback:
    def test_trims_layers_by_correct_amount(self) -> None:
        store = KVCacheStore()
        layer = _make_trimmable_layer(offset=10)
        store.put("c1", CacheEntry(cache=[layer], length=10))

        position = asyncio.run(rollback("c1", 6, store))

        assert position == 6
        layer.trim.assert_called_once_with(4)  # 10 - 6 = 4

    def test_updates_entry_length(self) -> None:
        store = KVCacheStore()
        layer = _make_trimmable_layer(offset=8)
        store.put("c1", CacheEntry(cache=[layer], length=8))

        asyncio.run(rollback("c1", 3, store))

        assert store.get("c1").length == 3  # type: ignore[union-attr]

    def test_rollback_to_zero(self) -> None:
        store = KVCacheStore()
        layer = _make_trimmable_layer(offset=5)
        store.put("c1", CacheEntry(cache=[layer], length=5))

        position = asyncio.run(rollback("c1", 0, store))

        assert position == 0
        layer.trim.assert_called_once_with(5)

    def test_rollback_to_current_position_is_noop(self) -> None:
        store = KVCacheStore()
        layer = _make_trimmable_layer(offset=7)
        store.put("c1", CacheEntry(cache=[layer], length=7))

        position = asyncio.run(rollback("c1", 7, store))

        assert position == 7
        layer.trim.assert_called_once_with(0)

    def test_raises_if_cache_not_found(self) -> None:
        store = KVCacheStore()
        with pytest.raises(ValueError, match="cache not found"):
            asyncio.run(rollback("missing", 0, store))

    def test_raises_if_position_negative(self) -> None:
        store = KVCacheStore()
        store.put("c1", CacheEntry(cache=[MagicMock()], length=5))
        with pytest.raises(ValueError, match="out of range"):
            asyncio.run(rollback("c1", -1, store))

    def test_raises_if_position_beyond_length(self) -> None:
        store = KVCacheStore()
        store.put("c1", CacheEntry(cache=[MagicMock()], length=5))
        with pytest.raises(ValueError, match="out of range"):
            asyncio.run(rollback("c1", 6, store))

    def test_raises_for_non_trimmable_layers(self) -> None:
        # A partial rollback (some layers trimmed, others not) would leave the
        # length counter disagreeing with actual per-layer offsets.  Better to
        # fail loudly so the problem surfaces at deployment time.
        store = KVCacheStore()
        layer = MagicMock()
        layer.is_trimmable.return_value = False
        store.put("c1", CacheEntry(cache=[layer], length=5))

        with pytest.raises(RuntimeError, match="does not support trim"):
            asyncio.run(rollback("c1", 3, store))

    def test_checkpoint_then_rollback_restores_position(self) -> None:
        """checkpoint + rollback round-trip."""
        store = KVCacheStore()
        layer = _make_trimmable_layer(offset=10)
        store.put("c1", CacheEntry(cache=[layer], length=10))

        saved = asyncio.run(checkpoint("c1", store))
        # Simulate additional tokens being added
        store.put("c1", CacheEntry(cache=[layer], length=15))

        asyncio.run(rollback("c1", saved, store))

        assert store.get("c1").length == saved  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# evict
# ---------------------------------------------------------------------------


class TestEvict:
    def test_removes_entry_from_store(self) -> None:
        store = KVCacheStore()
        store.put("c1", CacheEntry(cache=[MagicMock()], length=5))

        result = asyncio.run(evict("c1", store))

        assert result is True
        assert not store.has("c1")

    def test_raises_if_cache_not_found(self) -> None:
        store = KVCacheStore()
        with pytest.raises(ValueError, match="cache not found"):
            asyncio.run(evict("missing", store))

    def test_does_not_affect_other_entries(self) -> None:
        store = KVCacheStore()
        store.put("c1", CacheEntry(cache=[MagicMock()], length=5))
        store.put("c2", CacheEntry(cache=[MagicMock()], length=3))

        asyncio.run(evict("c1", store))

        assert not store.has("c1")
        assert store.has("c2")
