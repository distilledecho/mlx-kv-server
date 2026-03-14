"""Tests for KVCacheStore state transitions."""

from unittest.mock import MagicMock

from mlx_kv_server.cache import CacheEntry, KVCacheStore


def _entry(length: int = 0) -> CacheEntry:
    return CacheEntry(cache=[MagicMock()], length=length)


class TestKVCacheStore:
    def test_empty_store_has_nothing(self) -> None:
        store = KVCacheStore()
        assert not store.has("x")
        assert store.get("x") is None
        assert len(store) == 0

    def test_put_and_get(self) -> None:
        store = KVCacheStore()
        entry = _entry(length=3)
        store.put("abc", entry)
        assert store.has("abc")
        assert store.get("abc") is entry

    def test_put_replaces_existing(self) -> None:
        store = KVCacheStore()
        first = _entry(length=1)
        second = _entry(length=2)
        store.put("x", first)
        store.put("x", second)
        assert store.get("x") is second
        assert len(store) == 1

    def test_delete_existing_returns_true(self) -> None:
        store = KVCacheStore()
        store.put("y", _entry())
        assert store.delete("y") is True
        assert not store.has("y")

    def test_delete_absent_returns_false(self) -> None:
        store = KVCacheStore()
        assert store.delete("nope") is False

    def test_len_tracks_entries(self) -> None:
        store = KVCacheStore()
        store.put("a", _entry())
        store.put("b", _entry())
        assert len(store) == 2
        store.delete("a")
        assert len(store) == 1

    def test_multiple_cache_ids_are_independent(self) -> None:
        store = KVCacheStore()
        ea = _entry(length=10)
        eb = _entry(length=20)
        store.put("a", ea)
        store.put("b", eb)
        assert store.get("a") is ea
        assert store.get("b") is eb
        store.delete("a")
        assert store.get("b") is eb
