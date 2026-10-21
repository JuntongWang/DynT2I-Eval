from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "runtime.json"
EXAMPLE_CONFIG_PATH = REPO_ROOT / "configs" / "runtime.example.json"
PLACEHOLDER_PREFIXES = ("/path/to/", "<path-to-", "CHANGE_ME")


def get_repo_root() -> Path:
    return REPO_ROOT


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=4)
def load_runtime_config(config_path: str | None = None) -> dict[str, Any]:
    explicit = (
        config_path
        or os.environ.get("DYNT2I_EVAL_CONFIG")
        or os.environ.get("DYNAMIC_AIGI_BENCH_CONFIG")
    )
    candidate = Path(explicit).expanduser() if explicit else DEFAULT_CONFIG_PATH
    if explicit and not candidate.is_file():
        raise FileNotFoundError(f"Runtime configuration does not exist: {candidate}")
    if candidate.exists():
        return _load_json(candidate)
    if EXAMPLE_CONFIG_PATH.exists():
        return _load_json(EXAMPLE_CONFIG_PATH)
    return {}


def refresh_runtime_config() -> None:
    load_runtime_config.cache_clear()


def get_config_value(*keys: str, default: Any = None, config: dict[str, Any] | None = None) -> Any:
    current: Any = config if config is not None else load_runtime_config()
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def is_placeholder(value: str | None) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    return any(text.startswith(prefix) for prefix in PLACEHOLDER_PREFIXES)


def require_runtime_path(value: str | None, field_name: str) -> str:
    if is_placeholder(value):
        raise ValueError(
            f"Configuration value '{field_name}' is not set. "
            f"Populate configs/runtime.json or set DYNT2I_EVAL_CONFIG."
        )
    path = Path(str(value)).expanduser()
    return str(path if path.is_absolute() else REPO_ROOT / path)


def resolve_prompt_builder_model_path(
    override: str | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    if override:
        return override
    cfg = config if config is not None else load_runtime_config()
    candidate = get_config_value("prompt_builder_model_path", config=cfg)
    if candidate:
        return require_runtime_path(candidate, "prompt_builder_model_path")
    fallback = get_config_value("model_paths", "Qwen3.5-9B", config=cfg)
    return require_runtime_path(fallback, "model_paths.Qwen3.5-9B")
