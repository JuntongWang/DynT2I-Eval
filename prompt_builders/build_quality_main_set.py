#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from prompt_builders.common import ROOT
from benchmark.runtime import get_repo_root


GENERATOR = ROOT / "generate_quality_vnext.py"
ANALYZER = ROOT / "analyze_generated_outputs.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="qwen3.5")
    parser.add_argument("--logic-prob", type=float, default=0.03)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-root", default=str(ROOT / "output" / "quality_main"))
    return parser.parse_args()


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=get_repo_root())


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_file = output_root / "quality_main.jsonl"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    generator_cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        "-m",
        "prompt_builders.generate_quality_vnext",
        "--count",
        str(args.count),
        "--logic-prob",
        str(args.logic_prob),
        "--output",
        str(output_file),
    ]
    if args.model_path:
        generator_cmd.extend(["--model-path", args.model_path])
    if args.seed is not None:
        generator_cmd.extend(["--seed", str(args.seed)])
    run(generator_cmd, env=env)

    run(
        [
            sys.executable,
            "-m",
            "prompt_builders.analyze_generated_outputs",
            "--input",
            str(output_file),
            "--output-dir",
            str(output_root / "analysis"),
        ]
    )

    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}

    runs = manifest.get("runs", [])
    run_meta = {"count": args.count, "logic_prob": args.logic_prob, "seed": args.seed}
    runs.append(run_meta)
    manifest = {"merged": str(output_file), "num_runs": len(runs), "runs": runs}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
