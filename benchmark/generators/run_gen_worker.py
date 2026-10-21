from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.generators.factory import create_generator


def parse_args():
    parser = argparse.ArgumentParser(description="Image generation worker.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--gpu_id", required=True)
    parser.add_argument("--tasks_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ready_file", default="")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print("==================================================")
    print(f"[Worker] Model: {args.model_name}")
    print(f"[Worker] GPU binding: {args.gpu_id}")
    print("==================================================")

    with open(args.tasks_json, "r", encoding="utf-8") as handle:
        prompt_list = json.load(handle)

    try:
        generator = create_generator(args.model_name, args.output_dir, config_path=args.config)
        generator.generate_batch(prompt_list, ready_file=args.ready_file or None)
    except Exception as exc:
        print(f"[Worker] Fatal error while running {args.model_name}: {exc}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[Worker] {args.model_name} finished.\n")


if __name__ == "__main__":
    main()
