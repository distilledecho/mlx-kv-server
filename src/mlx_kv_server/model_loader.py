"""MLX model loading.

Wraps mlx_lm.load so the rest of the server never imports mlx_lm directly.
"""

from typing import Any

import mlx_lm


def load_model(model_name: str) -> tuple[Any, Any]:
    """Load an MLX model and tokenizer by name or local path.

    Args:
        model_name: HuggingFace repo id or local directory path.

    Returns:
        ``(model, tokenizer)`` as returned by mlx_lm.load.
    """
    return mlx_lm.load(model_name)  # type: ignore[no-any-return]
