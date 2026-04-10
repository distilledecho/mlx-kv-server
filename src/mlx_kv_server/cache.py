"""KV tensor cache store.

Maps cache_id → (mlx KV cache objects, token count).
All tensor operations remain in this module — callers hold opaque handles.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    """A single cached KV state."""

    cache: list[Any]  # mlx_lm KVCache objects, one per layer
    length: int  # number of tokens prefilled so far


class KVCacheStore:
    """In-memory store of KV cache entries keyed by cache_id."""

    def __init__(self) -> None:
        self._store: dict[str, CacheEntry] = {}

    def get(self, cache_id: str) -> CacheEntry | None:
        """Return the entry for *cache_id*, or None if absent."""
        return self._store.get(cache_id)

    def put(self, cache_id: str, entry: CacheEntry) -> None:
        """Insert or replace the entry for *cache_id*."""
        self._store[cache_id] = entry

    def delete(self, cache_id: str) -> bool:
        """Remove the entry for *cache_id*. Returns True if it existed."""
        if cache_id in self._store:
            del self._store[cache_id]
            return True
        return False

    def has(self, cache_id: str) -> bool:
        """Return True if *cache_id* is present."""
        return cache_id in self._store

    def total_tokens(self) -> int:
        """Return the total number of tokens across all active cache entries."""
        return sum(entry.length for entry in self._store.values())

    def __len__(self) -> int:
        return len(self._store)
