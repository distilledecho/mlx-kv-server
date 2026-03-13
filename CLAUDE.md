# CLAUDE.md — mlx-kv-server

> Instructions for Claude Code. Read this file before writing any code in this repo.
> For project-wide architecture context, design principles, and vision: see `daemon` repo — `CLAUDE.md` and `PRD.md`.

**At the start of every session, before writing any code:**

1. Read the issue you are working on, including all comments:
   ```bash
   gh issue view {number} --repo distilledecho/mlx-kv-server --comments
   ```
2. Read any ADR issues referenced in the issue body:
   ```bash
   gh issue view {adr-number} --repo distilledecho/daemon --comments
   ```

Issue comments are the living record of decisions made after the issue was filed. Always read them — do not rely on the issue body alone.

---

## What This Repo Is

A minimal bare metal MLX inference server for Apple Silicon. Owns exactly two things: a loaded model and a KV tensor cache in GPU memory. Exposes both via a Unix socket using a simple primitive interface.

This is the inference backend for `distilledecho/daemon`. It is **not** a general-purpose inference server — it is purpose-built for the KV cache architecture that enables low-TTFT persistent conversation.

**Key constraint:** No business logic, no RAG awareness, no scheduling. If you find yourself adding anything beyond model loading, cache management, and the five primitives, stop — that logic belongs in `daemon`.

---

## Hardware & Runtime

| Property | Value |
|---|---|
| Machine | Apple M1 Max, 32GB unified memory |
| OS | macOS (bare metal — this server never runs in a container) |
| Inference | MLX only |
| Python | 3.11+ |
| Package manager | uv exclusively |

**Never suggest CUDA, PyTorch GPU, or any non-MLX inference path.** This server exists specifically to keep KV tensors in Apple Silicon unified memory.

---

## The Socket API

The server binds to a Unix socket (path from config). The `daemon` container connects to this socket and issues requests using these five primitives — nothing else.

| Primitive | Signature | Description |
|---|---|---|
| `prefill` | `prefill(tokens, cache_id)` → handle | Extends KV cache with new tokens, returns a handle |
| `generate` | `generate(tokens, cache_id)` → token stream | Streams output tokens using existing cache |
| `checkpoint` | `checkpoint(cache_id)` → snapshot | Snapshots current cache state |
| `rollback` | `rollback(cache_id, position)` → restored | Restores cache to a prior position |
| `evict` | `evict(cache_id)` → freed | Frees GPU memory for a cache slot |

**The container never touches KV tensors directly.** Cache handles are opaque references. All tensor operations stay in this server.

---

## Project Scaffolding

Scaffolded from the [Diamond Light Source Python copier template](https://github.com/DiamondLightSource/python-copier-template):

- **src-layout:** All package code in `src/mlx_kv_server/`
- **`pyproject.toml`:** All dependencies and tool config. No `requirements.txt`.
- **uv:** All package management. Never use pip directly or create venvs manually.
- **Pre-commit:** ruff, pyright. Run `pre-commit run --all-files` before committing.
- **CI:** GitHub Actions. Do not create parallel CI workflows.

Do not fight the template.

---

## Coding Rules

Hard constraints:

1. **Async everywhere.** All socket handling, model calls, and cache operations must be `async`. No blocking calls on the main thread.
2. **MLX only.** No PyTorch, no CUDA, no cloud inference. All tensor operations use MLX.
3. **Config from file.** Socket path, model name, and cache settings come from a config file — never hardcoded.
4. **No telemetry.** Check new dependencies for telemetry and disable explicitly if found.
5. **No business logic.** No RAG, no scheduling, no conversation management. If it isn't model loading, cache management, or one of the five primitives, it doesn't belong here.
6. **Never containerized.** This server runs bare metal on macOS only. Do not add Dockerfile or container CI.

---

## Repository Structure

```
mlx-kv-server/
    pyproject.toml
    uv.lock
    CLAUDE.md
    .env.example          # MLX_KV_SOCKET_PATH, model name, etc.
    src/
        mlx_kv_server/
            __init__.py
            server.py         # Unix socket server, request dispatch
            cache.py          # KV tensor cache store: dict[cache_id → tensors]
            primitives.py     # prefill, generate, checkpoint, rollback, evict
            model_loader.py   # loads MLX model from config
    tests/
        test_cache.py
        test_primitives.py
        test_socket.py
    docs/
        research/             # deep-dive write-ups if warranted
```

---

## Testing Expectations

- Every primitive must have a corresponding test
- Cache operations (checkpoint, rollback, evict) must have tests verifying correct state transitions
- Socket tests should verify the server handles connection errors and malformed requests gracefully
- Tests run via `uv run pytest` — do not introduce a separate test runner
- Run `pre-commit run --all-files` before committing

---

## Git & GitHub Workflow

### Branch Naming

```
issue-{number}/{short-description}
```
Example: `issue-1/unix-socket-server`, `issue-2/checkpoint-rollback-evict`

Branch from `main`. One issue per branch.

### Commit Messages

Follow Conventional Commits:
```
type(scope): short description (#issue-number)
```
Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

### Pull Requests

Use the PR template. Every PR must include:
- What changed and why
- Which issue it closes
- How to test it
- Anything uncertain — flag for human review

**Do not merge your own PRs.**

### Board Management

Move issues on the project board as you work:
```bash
# When starting work
gh project item-edit ... # → In Progress

# When opening a PR
gh project item-edit ... # → In Review
```

---

## Open Questions

Before implementing anything that touches these areas, flag rather than assume:

- Eviction policy under GPU memory pressure (not yet designed)
- Persistent cache across server restarts (not in V1 scope)

---

## What This Server Is Not

- Not a general inference API (use LM Studio or mlx-lm serve for that)
- Not a multi-user server (single-user, single-daemon connection)
- Not containerized (bare metal macOS only, by design — see ADR #3 in `daemon` repo)
