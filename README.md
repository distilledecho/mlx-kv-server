[![CI](https://github.com/distilledecho/mlx-kv-server/actions/workflows/ci.yml/badge.svg)](https://github.com/distilledecho/mlx-kv-server/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/distilledecho/mlx-kv-server/branch/main/graph/badge.svg)](https://codecov.io/gh/distilledecho/mlx-kv-server)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# mlx_kv_server

A bare-metal MLX inference server for Apple Silicon with persistent KV tensor caching. The server loads a model once and exposes five primitives for cache management: `prefill`, `generate`, `checkpoint`, `rollback`, and `evict`. Designed as the inference backend for the `distilledecho/daemon` container, enabling low-TTFT persistent conversation with KV cache support.

Source          | <https://github.com/distilledecho/mlx-kv-server>
:---:           | :---:
Documentation   | <https://distilledecho.github.io/mlx-kv-server>
Releases        | <https://github.com/distilledecho/mlx-kv-server/releases>

## What mlx-kv-server Does

The server exposes five core primitives via a Unix socket:

| Primitive | Purpose |
|-----------|---------|
| **`prefill`** | Extend a KV cache with new tokens; returns a handle |
| **`generate`** | Stream output tokens using an existing KV cache |
| **`checkpoint`** | Snapshot the current state of a KV cache |
| **`rollback`** | Restore a KV cache to a prior checkpoint position |
| **`evict`** | Free GPU memory for a cache slot |

**Key features:**

- **No business logic** — purely model loading, KV cache management, and token I/O
- **Async I/O** — all socket handling and model operations are non-blocking
- **Apple Silicon only** — bare metal on macOS using MLX for inference
- **Persistent caching** — hold KV tensors in GPU memory across multiple inference calls
- **Low-TTFT inference** — reuse cache state to minimize time-to-first-token in conversation workflows

Cache handles are opaque references — the container never accesses KV tensors directly. All tensor operations stay in this server.

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Create a config file

```bash
cp config.toml.example config.toml
```

Edit `config.toml` to configure:
- **`socket_path`** — Unix socket path where the server listens (default: `/tmp/mlx-kv-server.sock`)
- **`model_name`** — HuggingFace model ID or local path (must be an MLX-compatible model). Example: `mlx-community/Qwen3.5-35B-A3B-4bit`
- **`max_tokens`** — Maximum tokens to generate per request (default: 512)
- **`temperature`** — Sampling temperature, 0.0 for deterministic (default: 0.0)

### 3. Run the server

```bash
uv run python -m mlx_kv_server
```

Or with a custom config path:

```bash
uv run python -m mlx_kv_server --config /path/to/config.toml
```

The server will:
- Load the model into GPU memory (first run downloads from HuggingFace)
- Create a Unix socket at the configured path
- Listen for requests from the daemon container

View the version:

```bash
uv run python -m mlx_kv_server --version
```

<!-- README only content. Anything below this line won't be included in index.md -->

See https://distilledecho.github.io/mlx-kv-server for more detailed documentation.
