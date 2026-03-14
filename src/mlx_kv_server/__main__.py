"""Interface for ``python -m mlx_kv_server``."""

import asyncio
import logging
from argparse import ArgumentParser
from collections.abc import Sequence

from . import __version__
from .cache import KVCacheStore
from .config import load_config
from .model_loader import load_model
from .server import run_server

__all__ = ["main"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main(args: Sequence[str] | None = None) -> None:
    """Argument parser for the CLI."""
    parser = ArgumentParser(
        prog="mlx-kv-server",
        description="Bare-metal MLX KV cache server for Apple Silicon.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=__version__,
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default="config.toml",
        help="Path to TOML config file (default: config.toml)",
    )
    parsed = parser.parse_args(args)

    config = load_config(parsed.config)
    model, tokenizer = load_model(config.model_name)
    cache_store = KVCacheStore()

    asyncio.run(run_server(config, model, tokenizer, cache_store))


if __name__ == "__main__":
    main()
