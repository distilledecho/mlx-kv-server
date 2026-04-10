"""Configuration loading from TOML file."""

import tomllib
from dataclasses import dataclass, field


@dataclass
class Config:
    """Server configuration."""

    socket_path: str
    model_name: str
    max_tokens: int = 512
    temperature: float = 0.0
    cache_capacity_tokens: int = 8192
    extra: dict[str, object] = field(default_factory=dict)


def load_config(path: str) -> Config:
    """Load configuration from a TOML file.

    Args:
        path: Path to the TOML config file.

    Returns:
        Populated Config instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If required keys are missing.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    return Config(
        socket_path=data["socket_path"],
        model_name=data["model_name"],
        max_tokens=int(data.get("max_tokens", 512)),
        temperature=float(data.get("temperature", 0.0)),
        cache_capacity_tokens=int(data.get("cache_capacity_tokens", 8192)),
    )
