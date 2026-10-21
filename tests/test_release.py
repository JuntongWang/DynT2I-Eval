from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from benchmark.base_scheduler import BaseDynamicBenchmarkScheduler
from benchmark.ranking_policy import (
    DEFAULT_RANKING_CONFIG,
    apply_bayesian_update,
    pairwise_win_probability,
    resolve_micro_batch,
    sync_model_record,
)
from benchmark.runtime import load_runtime_config, refresh_runtime_config
from prompt_builders.build_quality_main_set import run

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["SDXL-Base-1.0", "SD-3.5-Medium"]


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch):
    monkeypatch.delenv("DYNT2I_EVAL_CONFIG", raising=False)
    monkeypatch.delenv("DYNAMIC_AIGI_BENCH_CONFIG", raising=False)
    refresh_runtime_config()
    yield
    refresh_runtime_config()


class FinishedWorker:
    def __init__(self, returncode=0):
        self.returncode = returncode

    def poll(self):
        return self.returncode


class LocalScheduler(BaseDynamicBenchmarkScheduler):
    """Exercise scheduler IO and updates with tiny images and local outcomes."""
    worker_status = 0
    write_images = True
    invalid_outcome = False

    @property
    def dimension_name(self):
        return "Test"

    def dispatch_worker(self, env_name, script_path, args_list):
        model_name = args_list[args_list.index("--model_name") + 1]
        if self.write_images:
            model_dir = self.output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            for prompt_id in self.prompts_db:
                Image.new("RGB", (2, 2), "white").save(model_dir / f"{prompt_id}.png")
        return FinishedWorker(self.worker_status)

    def evaluate_task(self, task, img_a_path, img_b_path):
        assert Path(img_a_path).is_absolute()
        if self.invalid_outcome:
            return None
        return {**task, "winner": "A"}


def make_scheduler(tmp_path, **kwargs):
    prompts = tmp_path / "prompts.jsonl"
    if not prompts.exists():
        prompts.write_text('{"id": 1, "prompt": "A blue cup."}\n{"id": 2, "prompt": "A red ball."}\n')
    return LocalScheduler(
        prompts_file=str(prompts), output_dir=str(tmp_path / "outputs"),
        state_file=str(tmp_path / "state.json"), api_url="http://localhost/unused",
        model_names=MODELS, micro_batch_size=2, gpu_a="0", gpu_b="1", **kwargs,
    )


def test_successful_batch_and_resume(tmp_path):
    scheduler = make_scheduler(tmp_path)
    assert scheduler.run_micro_batch()
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["round"] == 1
    assert state["used_prompt_ids"] == ["1", "2"]
    assert all(record["matches"] == 2 for record in state["models"].values())
    assert sum(record["wins"] for record in state["models"].values()) == 2
    assert len(json.loads((scheduler.output_dir / "eval_results_round_1.json").read_text())) == 2
    resumed = make_scheduler(tmp_path)
    assert not resumed.run_micro_batch()
    assert resumed.state["round"] == 1


@pytest.mark.parametrize("failure", ["worker", "missing_images", "evaluator"])
def test_failed_round_keeps_prompts_and_ratings(tmp_path, failure):
    scheduler = make_scheduler(tmp_path, eval_retries=1)
    if failure == "worker":
        scheduler.worker_status = 9
    elif failure == "missing_images":
        scheduler.write_images = False
    else:
        scheduler.invalid_outcome = True
    before = json.loads(json.dumps(scheduler.state))
    with pytest.raises(RuntimeError):
        scheduler.run_micro_batch()
    assert scheduler.state == before
    assert not scheduler.state_file.exists()


def test_resume_excludes_unselected_models_from_scheduling(tmp_path):
    scheduler = make_scheduler(tmp_path)
    scheduler.state["models"]["FLUX.1-dev"] = sync_model_record({"matches": -1})
    scheduler.save_state()
    resumed = make_scheduler(tmp_path)
    assert set(resumed.select_next_pair()) == set(MODELS)


@pytest.mark.parametrize("text", [
    '{"id": 1, "prompt": "A cup."}\n{"id": 1, "prompt": "A ball."}\n',
    '{"id": 1, "prompt": ""}\n',
])
def test_unusable_prompt_streams_fail_before_generation(tmp_path, text):
    (tmp_path / "prompts.jsonl").write_text(text)
    with pytest.raises(ValueError):
        make_scheduler(tmp_path)


def test_config_precedence_and_missing_file(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical.json"
    legacy = tmp_path / "legacy.json"
    explicit = tmp_path / "explicit.json"
    for path in [canonical, legacy, explicit]:
        path.write_text(json.dumps({"source": path.stem}))
    monkeypatch.setenv("DYNAMIC_AIGI_BENCH_CONFIG", str(legacy))
    monkeypatch.setenv("DYNT2I_EVAL_CONFIG", str(canonical))
    assert load_runtime_config()["source"] == "canonical"
    assert load_runtime_config(str(explicit))["source"] == "explicit"
    with pytest.raises(FileNotFoundError):
        load_runtime_config(str(tmp_path / "missing.json"))


def test_scheduler_config_controls_api_default(tmp_path, monkeypatch):
    import text_alignment_benchmark

    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"api_urls": {"text_alignment": "http://localhost:9000/evaluate"}}))
    monkeypatch.setattr(sys, "argv", ["text_alignment_benchmark.py", "--config", str(path)])
    assert text_alignment_benchmark.parse_args().api_url == "http://localhost:9000/evaluate"


def test_quality_builder_subprocess_can_import_package(tmp_path):
    output = tmp_path / "analysis"
    run([
        sys.executable, "-m", "prompt_builders.analyze_generated_outputs",
        "--input", str(ROOT / "examples/quality.jsonl"), "--output-dir", str(output),
    ])
    assert json.loads((output / "summary.json").read_text())["count"] == 2


def test_bayesian_update_is_symmetric_and_uncertainty_is_bounded():
    winner, loser = sync_model_record({}), sync_model_record({})
    initial_sum = winner["mu"] + loser["mu"]
    apply_bayesian_update(winner, loser, weight=2)
    assert winner["mu"] > loser["mu"]
    assert winner["mu"] + loser["mu"] == pytest.approx(initial_sum)
    for _ in range(300):
        apply_bayesian_update(winner, loser, is_tie=True)
    for record in [winner, loser]:
        assert math.isfinite(record["mu"])
        assert record["sigma"] >= DEFAULT_RANKING_CONFIG.sigma_conv
    assert resolve_micro_batch(0, 0, 15)["macro_winner"] == "Tie"
    assert pairwise_win_probability(3, 0, 3, 0) == pytest.approx(0.5)


@pytest.mark.parametrize("entry", [
    "text_alignment_benchmark.py", "perceptual_quality_benchmark.py", "aesthetic_quality_benchmark.py",
    "build_text_alignment_set.py", "build_perceptual_quality_set.py", "scripts/check_setup.py",
])
def test_user_entry_help_requires_no_weights(entry):
    result = subprocess.run([sys.executable, str(ROOT / entry), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
