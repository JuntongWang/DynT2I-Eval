#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path


BAD_OUTPUT_PATTERNS = [
    re.compile(r"[\u4e00-\u9fff].*[A-Za-z].*[\u4e00-\u9fff]"),
    re.compile(r"[\ufffd]"),
    re.compile(r"\*\*"),
    re.compile(r"avia"),
    re.compile(r"иво"),
]

TEMPLATE_PATTERNS = {
    "frozen_fire": re.compile(r"frozen fire", re.I),
    "rusted_glass": re.compile(r"rusted glass", re.I),
    "pierce_through": re.compile(r"pierc\w+ through", re.I),
    "without_shattering": re.compile(r"without shatter", re.I),
    "hyper_realistic": re.compile(r"hyper-realistic", re.I),
    "cinematic": re.compile(r"cinematic", re.I),
    "volumetric": re.compile(r"volumetric", re.I),
    "masterpiece": re.compile(r"masterpiece", re.I),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def looks_broken(record: dict) -> bool:
    raw_output = record.get("raw_output", "")
    prompt = record.get("prompt", "")
    if not prompt.strip():
        return True
    return any(pattern.search(raw_output) for pattern in BAD_OUTPUT_PATTERNS)


def main():
    args = parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_lengths = [len(row.get("prompt", "").split()) for row in rows if row.get("prompt")]
    broken = [row for row in rows if looks_broken(row)]
    logic_counts = Counter(row.get("logic_pool") for row in rows if row.get("logic_pool"))
    difficulty_counts = Counter(row.get("difficulty") for row in rows if row.get("difficulty"))

    template_hits = {
        key: sum(1 for row in rows if pattern.search(row.get("prompt", "")))
        for key, pattern in TEMPLATE_PATTERNS.items()
    }

    summary = {
        "count": len(rows),
        "nonempty_prompt_rate": round(sum(1 for row in rows if row.get("prompt")) / max(len(rows), 1), 4),
        "broken_rate": round(len(broken) / max(len(rows), 1), 4),
        "mean_prompt_words": round(statistics.mean(prompt_lengths), 2) if prompt_lengths else 0,
        "logic_pool_counts": dict(logic_counts),
        "difficulty_counts": dict(difficulty_counts),
        "template_hits": template_hits,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(out_dir / "broken_examples.json", "w", encoding="utf-8") as handle:
        json.dump(broken[:20], handle, ensure_ascii=False, indent=2)

    lines = ["# Generated Output Analysis", ""]
    for key, value in summary.items():
        rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        lines.append(f"- {key}: {rendered}")
    with open(out_dir / "README.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(json.dumps({"summary": summary, "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
