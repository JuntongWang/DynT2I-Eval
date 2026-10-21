from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from benchmark.model_registry import DEFAULT_MODEL_ORDER, MODEL_SPECS, get_model_env
from benchmark.ranking_policy import (
    DEFAULT_RANKING_CONFIG,
    config_as_dict,
    conservative_score,
    opponent_selection_tuple,
    resolve_micro_batch,
    sync_model_record,
)
from benchmark.runtime import load_runtime_config


RANKING_CFG = DEFAULT_RANKING_CONFIG


class BaseDynamicBenchmarkScheduler(ABC):
    def __init__(
        self,
        *,
        prompts_file: str,
        output_dir: str,
        state_file: str,
        api_url: str,
        model_names: list[str] | None = None,
        micro_batch_size: int = 15,
        gpu_a: str = "0",
        gpu_b: str = "1",
        enable_gpu_guard: bool = False,
        gpu_guard_gb: int = 32,
        config_path: str | None = None,
        worker_timeout: float = 3600.0,
        eval_retries: int = 3,
    ):
        self.config = load_runtime_config(config_path)
        self.config_path = str(Path(config_path).expanduser().resolve()) if config_path else None
        self.worker_timeout = worker_timeout
        self.eval_retries = eval_retries
        self.model_names = list(DEFAULT_MODEL_ORDER if model_names is None else model_names)
        if len(set(self.model_names)) != len(self.model_names) or len(self.model_names) < 2:
            raise ValueError("Select at least two distinct models.")
        unknown = set(self.model_names) - MODEL_SPECS.keys()
        if unknown:
            raise ValueError(f"Unregistered models: {', '.join(sorted(unknown))}")
        if micro_batch_size < 1 or worker_timeout <= 0 or eval_retries < 1:
            raise ValueError("Micro-batch size, worker timeout, and evaluation attempts must be positive.")
        if str(gpu_a) == str(gpu_b):
            raise ValueError("Use different GPUs for the two generation workers.")
        self.prompts_file = str(prompts_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.api_url = api_url
        self.model_envs = {name: get_model_env(name, self.config) for name in self.model_names}
        self.micro_batch_size = micro_batch_size
        self.gpu_a = str(gpu_a)
        self.gpu_b = str(gpu_b)
        self.enable_gpu_guard = enable_gpu_guard
        self.gpu_guard_gb = gpu_guard_gb
        self.generator_worker = Path(__file__).resolve().parent / "generators" / "run_gen_worker.py"

        self.state = self.load_state()
        self.prompts_db = self.load_prompts()

    @property
    @abstractmethod
    def dimension_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def evaluate_task(self, task: dict, img_a_path: str, img_b_path: str) -> dict | None:
        raise NotImplementedError

    def format_result_log(self, result: dict) -> str:
        return f"winner={result.get('winner', 'Tie')}"

    def required_prompt_fields(self) -> tuple[str, ...]:
        return ("prompt",)

    def normalize_prompt_record(self, record: dict) -> dict:
        return record

    def parse_legacy_raw_output(self, raw_output: str) -> dict:
        result = {}

        prompt_match = re.search(r'"prompt"\s*:\s*"((?:\\.|[^"])*)"', raw_output, re.DOTALL)
        if prompt_match:
            try:
                result["prompt"] = json.loads(f"\"{prompt_match.group(1)}\"")
            except json.JSONDecodeError:
                result["prompt"] = prompt_match.group(1)

        checklist_match = re.search(r'"checklist"\s*:\s*\[(.*?)(?:\]\s*[,}]?|\Z)', raw_output, re.DOTALL)
        if checklist_match:
            items = re.findall(r'"((?:\\.|[^"])*)"', checklist_match.group(1), re.DOTALL)
            if items:
                checklist = []
                for item in items:
                    try:
                        checklist.append(json.loads(f"\"{item}\""))
                    except json.JSONDecodeError:
                        checklist.append(item)
                result["checklist"] = checklist

        return result

    def load_prompts(self) -> dict[str, dict]:
        prompts = {}
        with open(self.prompts_file, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row = self.normalize_prompt_record(row)
                prompt_id = str(row["id"])
                if prompt_id in prompts:
                    raise ValueError(f"Duplicate prompt id {prompt_id} at line {line_number}.")
                if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                    raise ValueError(f"Prompt {prompt_id} has no nonempty prompt text.")
                if "checklist" in self.required_prompt_fields():
                    checklist = row.get("checklist")
                    if not isinstance(checklist, list) or not checklist or any(
                        not isinstance(item, str) or not item.strip() for item in checklist
                    ):
                        raise ValueError(f"Prompt {prompt_id} requires a nonempty list of checklist questions.")
                prompts[prompt_id] = row
        if not prompts:
            raise ValueError(f"Prompt file is empty: {self.prompts_file}")
        return prompts

    def get_prompt_record(self, prompt_id: str) -> dict:
        prompt_id = str(prompt_id)
        record = self.prompts_db.get(prompt_id)
        required_fields = self.required_prompt_fields()
        if record is None or any(field not in record for field in required_fields):
            print(f"[{self.dimension_name}] Reloading prompt file after a missing field on prompt {prompt_id}.")
            self.prompts_db = self.load_prompts()
            record = self.prompts_db.get(prompt_id)
        if record is None:
            raise KeyError(f"Prompt id {prompt_id} is missing from {self.prompts_file}.")
        for field in required_fields:
            if field not in record:
                raise KeyError(f"Prompt id {prompt_id} is missing required field '{field}'.")
        return record

    def load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            for model_name in self.model_names:
                if model_name not in state["models"]:
                    state["models"][model_name] = sync_model_record({})
            for model_data in state["models"].values():
                sync_model_record(model_data)
            state["ranking_config"] = config_as_dict(RANKING_CFG)
            return state

        return {
            "round": 0,
            "models": {model_name: sync_model_record({}) for model_name in self.model_names},
            "history": [],
            "used_prompt_ids": [],
            "ranking_config": config_as_dict(RANKING_CFG),
        }

    def save_state(self) -> None:
        self.state["ranking_config"] = config_as_dict(RANKING_CFG)
        for model_data in self.state["models"].values():
            sync_model_record(model_data)
        temporary = self.state_file.with_name(self.state_file.name + ".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=2)
        temporary.replace(self.state_file)

    def dispatch_worker(self, env_name: str, script_path: Path, args_list: list[str]):
        cmd = ["conda", "run", "--no-capture-output", "-n", env_name, "python", "-u", str(script_path), *args_list]
        if self.config_path:
            cmd.extend(["--config", self.config_path])
        print(f"[{self.dimension_name}] Dispatching: {' '.join(cmd)}")
        return subprocess.Popen(cmd)

    def calculate_bayesian_update(self, winner: str, loser: str, is_tie: bool = False, weight: float = 1.0) -> None:
        winner_data = self.state["models"][winner]
        loser_data = self.state["models"][loser]

        from benchmark.ranking_policy import apply_bayesian_update

        apply_bayesian_update(winner_data, loser_data, RANKING_CFG, is_tie=is_tie, weight=weight)
        winner_data["match_history"][loser] = winner_data["match_history"].get(loser, 0) + 1
        loser_data["match_history"][winner] = loser_data["match_history"].get(winner, 0) + 1
        for model_name in [winner, loser]:
            model_data = self.state["models"][model_name]
            model_data["score"] = conservative_score(
                model_data["mu"],
                model_data["sigma"],
                model_data["matches"],
                RANKING_CFG,
            )

    def select_next_pair(self):
        active_models = [name for name in self.model_names if self.state["models"][name]["status"] == "active"]
        converged_models = [name for name in self.model_names if self.state["models"][name]["status"] == "converged"]

        if not active_models:
            return None

        pivot_model = min(active_models, key=lambda name: self.state["models"][name]["matches"])
        if len(active_models) == 1:
            if not converged_models:
                return None
            pivot_mu = self.state["models"][pivot_model]["mu"]
            sparring_partner = min(
                converged_models,
                key=lambda name: abs(self.state["models"][name]["mu"] - pivot_mu),
            )
            print(f"[{self.dimension_name}] Single active model fallback: {pivot_model} vs {sparring_partner}")
            return pivot_model, sparring_partner

        best_opponent = None
        best_tuple = (-float("inf"), -float("inf"), -float("inf"))
        total_rounds = max(1, self.state["round"])
        for opponent in active_models:
            if opponent == pivot_model:
                continue
            mu_pivot = self.state["models"][pivot_model]["mu"]
            mu_opp = self.state["models"][opponent]["mu"]
            n_iq = max(1, self.state["models"][pivot_model]["match_history"].get(opponent, 0))
            sigma_opp = self.state["models"][opponent]["sigma"]
            current_tuple = opponent_selection_tuple(
                mu_pivot,
                mu_opp,
                sigma_opp,
                total_rounds,
                n_iq,
                random.random(),
                RANKING_CFG,
            )
            if current_tuple > best_tuple:
                best_tuple = current_tuple
                best_opponent = opponent
        return pivot_model, best_opponent

    def assign_models_to_gpus(self, model_a: str, model_b: str) -> tuple[str, str]:
        return model_a, model_b

    def remember_gpu_assignment(self, model_a: str, model_b: str) -> None:
        del model_a, model_b

    def build_task_record(self, prompt_id: str, model_a: str, model_b: str, prompt_record: dict) -> dict:
        return {
            "prompt_id": prompt_id,
            "model_A_name": model_a,
            "model_B_name": model_b,
            "prompt": prompt_record["prompt"],
            "seed": 42 + self.state["round"],
            "image_a_path": str(self.output_dir / model_a / f"{prompt_id}.png"),
            "image_b_path": str(self.output_dir / model_b / f"{prompt_id}.png"),
        }

    def is_image_ready(self, image_path: str) -> bool:
        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            return False
        try:
            with Image.open(image_path) as image:
                image.verify()
            return True
        except Exception:
            return False

    def spawn_gpu_guard(self, gpu_id: str):
        if not self.enable_gpu_guard:
            return None
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        command = [
            "python",
            "-c",
            (
                "import time, torch; "
                f"hold=torch.empty(({int(self.gpu_guard_gb)}, 1024, 1024, 1024), dtype=torch.uint8, device='cuda'); "
                "time.sleep(36000)"
            ),
        ]
        return subprocess.Popen(command, env=env)

    def record_history_snapshot(
        self,
        *,
        model_a: str,
        model_b: str,
        wins_a: int,
        wins_b: int,
        ties: int,
        batch_summary: dict,
    ) -> None:
        snapshot = {
            "round": self.state["round"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dimension": self.dimension_name,
            "matchup": {
                "model_a": model_a,
                "model_b": model_b,
                "winner": batch_summary["macro_winner"],
                "wins_a": wins_a,
                "wins_b": wins_b,
                "ties": ties,
                "p_a": round(batch_summary["p_a"], 4) if batch_summary["p_a"] is not None else None,
                "batch_weight": round(batch_summary["batch_weight"], 4),
            },
            "leaderboard": {
                model_name: {
                    "rank": 0,
                    "score": round(model_data["score"], 2),
                    "mu": round(model_data["mu"], 2),
                    "sigma": round(model_data["sigma"], 2),
                    "matches": model_data["matches"],
                    "win_rate": round(model_data["wins"] / max(1, model_data["matches"]), 3),
                    "status": model_data["status"],
                }
                for model_name, model_data in self.state["models"].items()
            },
        }

        sorted_models = sorted(snapshot["leaderboard"].items(), key=lambda item: item[1]["score"], reverse=True)
        for rank, (model_name, _) in enumerate(sorted_models, start=1):
            snapshot["leaderboard"][model_name]["rank"] = rank

        history_file = self.output_dir / "score_history.jsonl"
        with open(history_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

    def run_micro_batch(self) -> bool:
        pair = self.select_next_pair()
        if not pair:
            print(f"[{self.dimension_name}] All models have converged. Stopping.")
            return False

        model_a, model_b = self.assign_models_to_gpus(*pair)
        used_set = set(self.state["used_prompt_ids"])
        available_ids = [prompt_id for prompt_id in self.prompts_db.keys() if prompt_id not in used_set]
        if len(available_ids) < self.micro_batch_size:
            print(f"[{self.dimension_name}] Prompt pool is low. Reloading prompt file.")
            self.prompts_db = self.load_prompts()
            available_ids = [prompt_id for prompt_id in self.prompts_db.keys() if prompt_id not in used_set]

        batch_ids = available_ids[: self.micro_batch_size]
        if not batch_ids:
            print(f"[{self.dimension_name}] Prompt pool exhausted.")
            return False

        self.state["round"] += 1
        print(f"\n=== {self.dimension_name} | Round {self.state['round']} | {model_a} vs {model_b} ===")

        tasks = []
        for prompt_id in batch_ids:
            prompt_record = self.get_prompt_record(prompt_id)
            tasks.append(self.build_task_record(prompt_id, model_a, model_b, prompt_record))

        tasks_file = self.output_dir / f"tasks_round_{self.state['round']}.json"
        with open(tasks_file, "w", encoding="utf-8") as handle:
            json.dump(tasks, handle)

        workers = []
        guards = {}
        results_by_id = {}
        attempts = {task["prompt_id"]: 0 for task in tasks}
        pending = {}
        started = time.monotonic()
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            for model_name, gpu_id in [(model_a, self.gpu_a), (model_b, self.gpu_b)]:
                workers.append(self.dispatch_worker(
                    env_name=self.model_envs[model_name],
                    script_path=self.generator_worker,
                    args_list=[
                        "--model_name", model_name,
                        "--gpu_id", gpu_id,
                        "--tasks_json", str(tasks_file),
                        "--output_dir", str(self.output_dir),
                    ],
                ))
            self.remember_gpu_assignment(model_a, model_b)
            print(f"[{self.dimension_name}] Waiting for generation and evaluation...")

            while len(results_by_id) < len(tasks) or any(worker.poll() is None for worker in workers):
                if time.monotonic() - started > self.worker_timeout:
                    raise TimeoutError(f"Round exceeded {self.worker_timeout:g} seconds.")
                for index, worker in enumerate(workers):
                    status = worker.poll()
                    if status is not None and status != 0:
                        raise RuntimeError(f"Generation worker {index + 1} exited with status {status}.")
                    if status == 0 and index not in guards:
                        guards[index] = self.spawn_gpu_guard([self.gpu_a, self.gpu_b][index])

                for prompt_id, future in list(pending.items()):
                    if not future.done():
                        continue
                    del pending[prompt_id]
                    try:
                        result = future.result()
                        if result is None or result.get("winner") not in {"A", "B", "Tie"}:
                            raise ValueError("Evaluator returned no valid pairwise outcome.")
                    except Exception as exc:
                        if attempts[prompt_id] >= self.eval_retries:
                            raise RuntimeError(
                                f"Evaluation failed for prompt {prompt_id} after {attempts[prompt_id]} attempts."
                            ) from exc
                        print(f"[{self.dimension_name}] Retrying prompt {prompt_id}: {exc}")
                    else:
                        results_by_id[prompt_id] = result
                        print(f"[{self.dimension_name}] Prompt {prompt_id}: {self.format_result_log(result)}")

                generators_done = all(worker.poll() == 0 for worker in workers)
                for task in tasks:
                    prompt_id = task["prompt_id"]
                    if prompt_id in results_by_id or prompt_id in pending:
                        continue
                    img_a = os.path.abspath(task["image_a_path"])
                    img_b = os.path.abspath(task["image_b_path"])
                    if self.is_image_ready(img_a) and self.is_image_ready(img_b):
                        attempts[prompt_id] += 1
                        pending[prompt_id] = executor.submit(
                            self.evaluate_task, dict(task), img_a, img_b,
                        )
                    elif generators_done:
                        raise RuntimeError(f"Generation finished without valid images for prompt {prompt_id}.")
                if len(results_by_id) < len(tasks) or any(worker.poll() is None for worker in workers):
                    time.sleep(1.0)
        except BaseException:
            # Keep the round and prompt usage unchanged so the batch can be retried.
            self.state["round"] -= 1
            raise
        finally:
            for process in [*workers, *[guard for guard in guards.values() if guard is not None]]:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            executor.shutdown(wait=True, cancel_futures=True)

        results = [results_by_id[prompt_id] for prompt_id in batch_ids]

        eval_results_file = self.output_dir / f"eval_results_round_{self.state['round']}.json"
        with open(eval_results_file, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)

        wins_a = 0
        wins_b = 0
        ties = 0
        for result in results:
            winner_flag = result.get("winner", "Tie")
            if winner_flag == "A":
                wins_a += 1
            elif winner_flag == "B":
                wins_b += 1
            else:
                ties += 1

        total_valid = wins_a + wins_b + ties
        batch_summary = resolve_micro_batch(wins_a, wins_b, ties, RANKING_CFG)
        if total_valid > 0:
            self.state["models"][model_a]["wins"] += wins_a
            self.state["models"][model_a]["losses"] += wins_b
            self.state["models"][model_a]["ties"] += ties
            self.state["models"][model_b]["wins"] += wins_b
            self.state["models"][model_b]["losses"] += wins_a
            self.state["models"][model_b]["ties"] += ties
            self.state["models"][model_a]["matches"] += total_valid
            self.state["models"][model_b]["matches"] += total_valid

            if batch_summary["macro_winner"] == "A":
                self.calculate_bayesian_update(model_a, model_b, weight=batch_summary["batch_weight"])
            elif batch_summary["macro_winner"] == "B":
                self.calculate_bayesian_update(model_b, model_a, weight=batch_summary["batch_weight"])
            else:
                self.calculate_bayesian_update(model_a, model_b, is_tie=True, weight=batch_summary["batch_weight"])

        self.state["used_prompt_ids"].extend(batch_ids)
        self.record_history_snapshot(
            model_a=model_a,
            model_b=model_b,
            wins_a=wins_a,
            wins_b=wins_b,
            ties=ties,
            batch_summary=batch_summary,
        )
        self.save_state()
        return True
