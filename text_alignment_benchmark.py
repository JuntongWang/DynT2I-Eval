#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

from benchmark.base_scheduler import BaseDynamicBenchmarkScheduler
from benchmark.runtime import get_config_value, get_repo_root, load_runtime_config
from benchmark.model_registry import DEFAULT_MODEL_ORDER


REPO_ROOT = get_repo_root()


class TextAlignmentScheduler(BaseDynamicBenchmarkScheduler):
    @property
    def dimension_name(self) -> str:
        return "Text Alignment"

    def required_prompt_fields(self) -> tuple[str, ...]:
        return ("prompt", "checklist")

    def normalize_prompt_record(self, record: dict) -> dict:
        if "prompt" in record and "checklist" in record:
            return record
        raw_output = record.get("raw_output")
        if not raw_output:
            return record
        parsed = raw_output if isinstance(raw_output, dict) else self.parse_legacy_raw_output(raw_output)
        if isinstance(parsed, dict):
            if "prompt" in parsed and "prompt" not in record:
                record["prompt"] = parsed["prompt"]
            if "checklist" in parsed and "checklist" not in record:
                record["checklist"] = parsed["checklist"]
        return record

    def build_task_record(self, prompt_id: str, model_a: str, model_b: str, prompt_record: dict) -> dict:
        task = super().build_task_record(prompt_id, model_a, model_b, prompt_record)
        task["checklist"] = prompt_record["checklist"]
        return task

    def evaluate_task(self, task: dict, img_a_path: str, img_b_path: str) -> dict | None:
        start_time = time.time()
        payload = {
            "image_a_path": img_a_path,
            "image_b_path": img_b_path,
            "prompt_text": task["prompt"],
            "checklist": task["checklist"],
        }
        response = requests.post(self.api_url, json=payload, timeout=60)
        if response.status_code != 200:
            print(f"[Text Alignment] Evaluator returned {response.status_code}: {response.text[:200]}")
            return None
        result = response.json()
        task.update(result)
        task["eval_latency_seconds"] = round(time.time() - start_time, 2)
        return task

    def format_result_log(self, result: dict) -> str:
        return f"winner={result.get('winner', 'Tie')}"


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()
    config = load_runtime_config(config_args.config)
    parser = argparse.ArgumentParser(description="Run the Text Alignment benchmark scheduler.")
    parser.add_argument(
        "--prompts",
        default=str(REPO_ROOT / "prompt_builders" / "output" / "alignment_main" / "alignment_main.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "benchmark_outputs" / "text_alignment"),
    )
    parser.add_argument(
        "--state-file",
        default=str(REPO_ROOT / "benchmark_states" / "text_alignment.json"),
    )
    parser.add_argument(
        "--api-url",
        default=get_config_value("api_urls", "text_alignment", default="http://127.0.0.1:8000/evaluate", config=config),
    )
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODEL_ORDER, default=None)
    parser.add_argument("--worker-timeout", type=float, default=3600, help="Maximum seconds per round.")
    parser.add_argument("--eval-retries", type=int, default=3, help="Maximum evaluation attempts per prompt.")
    parser.add_argument("--micro-batch-size", type=int, default=15)
    parser.add_argument("--gpu-a", default="0")
    parser.add_argument("--gpu-b", default="1")
    parser.add_argument("--enable-gpu-guard", action="store_true")
    parser.add_argument("--gpu-guard-gb", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=0, help="Run a fixed number of rounds. Zero means until completion.")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    scheduler = TextAlignmentScheduler(
        prompts_file=args.prompts,
        output_dir=args.output_dir,
        state_file=args.state_file,
        api_url=args.api_url,
        micro_batch_size=args.micro_batch_size,
        gpu_a=args.gpu_a,
        gpu_b=args.gpu_b,
        enable_gpu_guard=args.enable_gpu_guard,
        gpu_guard_gb=args.gpu_guard_gb,
        config_path=args.config,
        model_names=args.models,
        worker_timeout=args.worker_timeout,
        eval_retries=args.eval_retries,
    )

    completed = 0
    while True:
        if args.rounds and completed >= args.rounds:
            break
        if not scheduler.run_micro_batch():
            break
        completed += 1


if __name__ == "__main__":
    main()
