#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time

import requests

from benchmark.base_scheduler import BaseDynamicBenchmarkScheduler
from benchmark.ranking_policy import pairwise_win_probability, winner_from_probability
from benchmark.runtime import get_config_value, get_repo_root, load_runtime_config
from benchmark.model_registry import DEFAULT_MODEL_ORDER


REPO_ROOT = get_repo_root()


class PerceptualQualityScheduler(BaseDynamicBenchmarkScheduler):
    def __init__(self, *args, iqa_samples: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.iqa_samples = iqa_samples

    @property
    def dimension_name(self) -> str:
        return "Perceptual Quality"

    def evaluate_task(self, task: dict, img_a_path: str, img_b_path: str) -> dict | None:
        start_time = time.time()
        response_a = requests.post(
            self.api_url,
            json={"image_path": img_a_path, "k_samples": self.iqa_samples},
            timeout=120,
        )
        response_b = requests.post(
            self.api_url,
            json={"image_path": img_b_path, "k_samples": self.iqa_samples},
            timeout=120,
        )
        if response_a.status_code != 200 or response_b.status_code != 200:
            print(
                f"[Perceptual Quality] Evaluator error. "
                f"A={response_a.status_code} B={response_b.status_code}"
            )
            return None

        data_a = response_a.json()
        data_b = response_b.json()
        mu_a = data_a.get("score", -1.0)
        var_a = data_a.get("variance", 0.0)
        mu_b = data_b.get("score", -1.0)
        var_b = data_b.get("variance", 0.0)
        if mu_a == -1.0 or mu_b == -1.0:
            return None

        prob_a_wins = pairwise_win_probability(mu_a, var_a, mu_b, var_b)
        task["score_A"] = round(mu_a, 4)
        task["var_A"] = round(var_a, 4)
        task["std_A"] = round(data_a.get("std", math.sqrt(max(var_a, 0.0))), 4)
        task["reasonings_A"] = data_a.get("reasonings", [])
        task["raw_scores_A"] = data_a.get("raw_scores", [])
        task["score_B"] = round(mu_b, 4)
        task["var_B"] = round(var_b, 4)
        task["std_B"] = round(data_b.get("std", math.sqrt(max(var_b, 0.0))), 4)
        task["reasonings_B"] = data_b.get("reasonings", [])
        task["raw_scores_B"] = data_b.get("raw_scores", [])
        task["prob_A_wins"] = round(prob_a_wins, 4)
        task["winner"] = winner_from_probability(prob_a_wins)
        task["eval_latency_seconds"] = round(time.time() - start_time, 2)
        return task

    def format_result_log(self, result: dict) -> str:
        return (
            f"winner={result.get('winner', 'Tie')} "
            f"score_A={result.get('score_A', -1):.2f} "
            f"score_B={result.get('score_B', -1):.2f}"
        )


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()
    config = load_runtime_config(config_args.config)
    parser = argparse.ArgumentParser(description="Run the Perceptual Quality benchmark scheduler.")
    parser.add_argument(
        "--prompts",
        default=str(REPO_ROOT / "prompt_builders" / "output" / "quality_main" / "quality_main.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "benchmark_outputs" / "perceptual_quality"),
    )
    parser.add_argument(
        "--state-file",
        default=str(REPO_ROOT / "benchmark_states" / "perceptual_quality.json"),
    )
    parser.add_argument(
        "--api-url",
        default=get_config_value(
            "api_urls",
            "perceptual_quality",
            default="http://127.0.0.1:8001/evaluate_iqa",
            config=config,
        ),
    )
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODEL_ORDER, default=None)
    parser.add_argument("--worker-timeout", type=float, default=3600, help="Maximum seconds per round.")
    parser.add_argument("--eval-retries", type=int, default=3, help="Maximum evaluation attempts per prompt.")
    parser.add_argument("--micro-batch-size", type=int, default=15)
    parser.add_argument("--gpu-a", default="0")
    parser.add_argument("--gpu-b", default="1")
    parser.add_argument("--iqa-samples", type=int, default=4)
    parser.add_argument("--enable-gpu-guard", action="store_true")
    parser.add_argument("--gpu-guard-gb", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    scheduler = PerceptualQualityScheduler(
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
        iqa_samples=args.iqa_samples,
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
