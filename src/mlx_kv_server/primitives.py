"""prefill and generate primitives.

All MLX / mlx_lm calls are synchronous and run inside asyncio.to_thread or a
dedicated threading loop so they never block the event loop.
"""

import asyncio
import threading
from collections.abc import AsyncGenerator
from typing import Any

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

from .cache import CacheEntry, KVCacheStore

# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------


def _sync_prefill(
    tokens: list[int],
    model: Any,
    prompt_cache: list[Any],
) -> None:
    """Run a forward pass to fill *prompt_cache* with KV state for *tokens*.

    Blocks until MLX has evaluated the cache (materialises the lazy graph).
    """
    prompt = mx.array([tokens])
    model(prompt, cache=prompt_cache)
    mx.eval(prompt_cache)


async def prefill(
    tokens: list[int],
    cache_id: str,
    model: Any,
    cache_store: KVCacheStore,
    lock: asyncio.Lock,
) -> str:
    """Extend the KV cache for *cache_id* with *tokens*.

    Creates a fresh cache entry if *cache_id* is not yet known.

    Args:
        tokens: Token IDs to prefill.
        cache_id: Opaque cache handle (created on first use).
        model: Loaded MLX model.
        cache_store: Shared KV cache store.
        lock: Async lock serialising model access.

    Returns:
        *cache_id* — the caller's opaque handle.
    """
    entry = cache_store.get(cache_id)
    if entry is None:
        prompt_cache = make_prompt_cache(model)
        length = 0
    else:
        prompt_cache = entry.cache
        length = entry.length

    async with lock:
        await asyncio.to_thread(_sync_prefill, tokens, model, prompt_cache)

    cache_store.put(
        cache_id, CacheEntry(cache=prompt_cache, length=length + len(tokens))
    )
    return cache_id


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


async def generate(
    tokens: list[int],
    cache_id: str,
    model: Any,
    tokenizer: Any,
    cache_store: KVCacheStore,
    lock: asyncio.Lock,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> AsyncGenerator[int, None]:
    """Stream output tokens using the cached KV state for *cache_id*.

    The *tokens* are used as the prompt for this generation turn (prefilled
    into the cache by generate_step), then new tokens are streamed until EOS
    or *max_tokens* is reached.

    Args:
        tokens: Prompt token IDs for this generation turn.
        cache_id: Opaque cache handle (must already exist).
        model: Loaded MLX model.
        tokenizer: Loaded tokenizer (provides eos_token_id).
        cache_store: Shared KV cache store.
        lock: Async lock serialising model access.
        max_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).

    Yields:
        Generated token IDs, one at a time.

    Raises:
        ValueError: If *cache_id* is not found in the store.
    """
    entry = cache_store.get(cache_id)
    if entry is None:
        raise ValueError(f"cache not found: {cache_id!r}")

    prompt_cache = entry.cache
    prompt = mx.array(tokens)
    eos_id: int = tokenizer.eos_token_id

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[int | Exception | None] = asyncio.Queue()

    sampler: Any
    if temperature == 0.0:
        sampler = lambda logprobs: mx.argmax(logprobs, axis=-1)  # noqa: E731
    else:
        import mlx_lm.sample_utils as su

        sampler = su.make_sampler(temp=temperature)

    def _run_generation() -> None:
        try:
            count = 0
            for token, _ in generate_step(
                prompt,
                model,
                prompt_cache=prompt_cache,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                token_int = int(token)
                loop.call_soon_threadsafe(queue.put_nowait, token_int)
                count += 1
                if token_int == eos_id or count >= max_tokens:
                    break
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async with lock:
        thread = threading.Thread(target=_run_generation, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        thread.join()

    # Update cached length (original + prompt tokens for this turn)
    cache_store.put(
        cache_id,
        CacheEntry(cache=prompt_cache, length=entry.length + len(tokens)),
    )
