from __future__ import annotations

import json
import re
from pathlib import Path

from benchmark.runtime import resolve_prompt_builder_model_path


ROOT = Path(__file__).resolve().parent
RESOURCE_DIR = ROOT / "resources"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_resource(name: str):
    return load_json(RESOURCE_DIR / name)


def extract_json_from_output(text: str):
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        raw = match.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        raw = match.group(0)
    try:
        return json.loads(raw)
    except Exception:
        return None


def latest_id(path: Path) -> int:
    if not path.exists():
        return 0
    latest = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("id"), int):
                latest = max(latest, row["id"])
    return latest


def resolve_builder_model_path(override: str | None) -> str:
    return resolve_prompt_builder_model_path(override)
