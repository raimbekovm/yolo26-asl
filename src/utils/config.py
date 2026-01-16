"""Configuration loading and management utilities."""

from pathlib import Path
from typing import Any, Optional, Union

import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.utils.constants import CONFIG_DIR


def load_config(
    config_name: str = "config",
    config_dir: Optional[Path] = None,
    overrides: Optional[list[str]] = None,
) -> DictConfig:
    """
    Load configuration using OmegaConf.

    Args:
        config_name: Name of the config file (without .yaml extension).
        config_dir: Directory containing config files.
        overrides: List of override strings (e.g., ["training.epochs=100"]).

    Returns:
        DictConfig: Merged configuration object.

    Example:
        >>> cfg = load_config("config", overrides=["training.epochs=50"])
        >>> print(cfg.training.epochs)
        50
    """
    if config_dir is None:
        config_dir = CONFIG_DIR

    config_path = config_dir / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load base config
    cfg = OmegaConf.load(config_path)

    # Resolve defaults if present
    if "defaults" in cfg:
        cfg = _resolve_defaults(cfg, config_dir)

    # Apply overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Resolve interpolations
    OmegaConf.resolve(cfg)

    logger.debug(f"Loaded config from {config_path}")
    return cfg


def _resolve_defaults(cfg: DictConfig, config_dir: Path) -> DictConfig:
    """Resolve default configs similar to Hydra."""
    defaults = cfg.pop("defaults", [])
    merged = OmegaConf.create()

    for default in defaults:
        if isinstance(default, str):
            if default == "_self_":
                merged = OmegaConf.merge(merged, cfg)
            else:
                default_cfg = _load_default(default, config_dir)
                merged = OmegaConf.merge(merged, default_cfg)

        elif isinstance(default, dict):
            for key, value in default.items():
                if key.startswith("override "):
                    # Handle override syntax
                    key = key.replace("override ", "")
                if value is not None:
                    default_path = config_dir / key / f"{value}.yaml"
                    if default_path.exists():
                        default_cfg = OmegaConf.load(default_path)
                        merged = OmegaConf.merge(merged, {key: default_cfg})

    # Merge remaining cfg (if _self_ wasn't in defaults)
    if cfg:
        merged = OmegaConf.merge(merged, cfg)

    return merged


def _load_default(name: str, config_dir: Path) -> DictConfig:
    """Load a single default config file."""
    # Try direct path first
    path = config_dir / f"{name}.yaml"
    if path.exists():
        return OmegaConf.load(path)

    # Try subdirectory
    parts = name.split("/")
    if len(parts) == 2:
        path = config_dir / parts[0] / f"{parts[1]}.yaml"
        if path.exists():
            return OmegaConf.load(path)

    logger.warning(f"Default config not found: {name}")
    return OmegaConf.create()


def load_yaml(path: Union[str, Path]) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    path = Path(path)
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data: Union[dict, DictConfig], path: Union[str, Path]) -> None:
    """Save data to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, DictConfig):
        data = OmegaConf.to_container(data, resolve=True)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.debug(f"Saved config to {path}")


def print_config(cfg: DictConfig, resolve: bool = True) -> None:
    """Pretty print configuration."""
    from rich import print as rprint
    from rich.syntax import Syntax

    yaml_str = OmegaConf.to_yaml(cfg, resolve=resolve)
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
    rprint(syntax)


def get_config_diff(cfg1: DictConfig, cfg2: DictConfig) -> dict[str, Any]:
    """Get difference between two configs."""
    diff = {}

    def _compare(d1: Any, d2: Any, path: str = "") -> None:
        if isinstance(d1, DictConfig) and isinstance(d2, DictConfig):
            all_keys = set(d1.keys()) | set(d2.keys())
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                v1 = d1.get(key)
                v2 = d2.get(key)
                _compare(v1, v2, new_path)
        elif d1 != d2:
            diff[path] = {"old": d1, "new": d2}

    _compare(cfg1, cfg2)
    return diff
