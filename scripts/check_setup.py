#!/usr/bin/env python3
"""Check selected runtime paths without importing models or using GPUs."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.model_registry import DEFAULT_MODEL_ORDER, get_model_env
from benchmark.runtime import get_config_value, load_runtime_config, require_runtime_path

JUDGES = {
    "text_alignment": ("Qwen3.5-9B", None),
    "perceptual_quality": ("VisualQuality-R1-7B", "visualquality_r1_root"),
    "aesthetic_quality": ("ArtiMuse", "artimuse_root"),
}


def check(config: dict, dimension: str, model_names: list[str], builder: bool = False) -> list[str]:
    errors = []
    paths = [("model_paths", name) for name in model_names]
    judge, source = JUDGES[dimension]
    paths.append(("model_paths", judge))
    if source:
        paths.append(("external_code", source))
    if "InternVL-U" in model_names:
        paths.append(("external_code", "internvl_u_root"))
    if builder:
        paths.append(("prompt_builder_model_path",))
    if len(model_names) < 2 or len(set(model_names)) != len(model_names):
        errors.append("Select at least two distinct generation models.")
    for keys in paths:
        field = ".".join(keys)
        try:
            resolved = require_runtime_path(get_config_value(*keys, config=config), field)
            if not Path(resolved).is_dir():
                errors.append(f"{field}: directory does not exist: {resolved}")
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--dimension", choices=JUDGES, required=True)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODEL_ORDER, default=DEFAULT_MODEL_ORDER)
    parser.add_argument("--include-builder", action="store_true")
    args = parser.parse_args()
    try:
        config = load_runtime_config(args.config)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Configuration error: {exc}\n")
    errors = check(config, args.dimension, args.models, args.include_builder)
    if not shutil.which("conda"):
        errors.append("conda is not available on PATH; generation workers require it.")
    print(f"Dimension: {args.dimension}")
    for name in args.models:
        print(f"  {name}: conda environment {get_model_env(name, config)}")
    if errors:
        print("\nSetup needs attention:")
        for message in errors:
            print(f"  - {message}")
        raise SystemExit(1)
    print("\nConfigured directories exist and conda is available.")
    print("Checkpoint contents, Conda environments, imports, GPU capacity, and running services still require runtime verification.")


if __name__ == "__main__":
    main()
