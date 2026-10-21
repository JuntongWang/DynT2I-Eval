#!/usr/bin/env python3
"""Demonstrate online ranking with synthetic outcomes; no models are loaded."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.ranking_policy import (
    apply_bayesian_update,
    opponent_selection_tuple,
    resolve_micro_batch,
    sync_model_record,
)


def simulate(rounds: int, seed: int, micro_batch_size: int = 15) -> tuple[dict, list]:
    rng = random.Random(seed)
    strengths = {"Example-A": 0.8, "Example-B": 0.3, "Example-C": -0.2, "Example-D": -0.7}
    models = {name: sync_model_record({}) for name in strengths}
    history = []
    for round_id in range(1, rounds + 1):
        pivot = min(models, key=lambda name: models[name]["matches"])
        opponent = max(
            (name for name in models if name != pivot),
            key=lambda name: opponent_selection_tuple(
                models[pivot]["mu"], models[name]["mu"], models[name]["sigma"],
                round_id - 1, models[pivot]["match_history"].get(name, 0), rng.random(),
            ),
        )
        prob_a = 1.0 / (1.0 + math.exp(-(strengths[pivot] - strengths[opponent])))
        wins_a = wins_b = ties = 0
        for _ in range(micro_batch_size):
            if rng.random() < 0.1:
                ties += 1
            elif rng.random() < prob_a:
                wins_a += 1
            else:
                wins_b += 1
        batch = resolve_micro_batch(wins_a, wins_b, ties)
        for name, other, wins, losses in [
            (pivot, opponent, wins_a, wins_b), (opponent, pivot, wins_b, wins_a),
        ]:
            record = models[name]
            record["matches"] += micro_batch_size
            record["wins"] += wins
            record["losses"] += losses
            record["ties"] += ties
            record["match_history"][other] = record["match_history"].get(other, 0) + 1
        winner, loser = (opponent, pivot) if batch["macro_winner"] == "B" else (pivot, opponent)
        apply_bayesian_update(
            models[winner], models[loser], is_tie=batch["is_tie"], weight=batch["batch_weight"],
        )
        for record in models.values():
            sync_model_record(record)
        history.append({"round": round_id, "model_a": pivot, "model_b": opponent, **batch})
    return models, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "demo_outputs")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    models, history = simulate(args.rounds, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"synthetic": True, "seed": args.seed, "rounds": args.rounds, "models": models, "history": history}
    output = args.output_dir / "synthetic_ranking.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Synthetic demonstration only: these are not text-to-image model results.\n")
    print(f"{'Rank':>4}  {'Model':<14} {'Score':>9} {'Mean':>9} {'Sigma':>9} {'Matches':>8}")
    for rank, (name, record) in enumerate(sorted(models.items(), key=lambda row: -row[1]["score"]), 1):
        print(f"{rank:4}  {name:<14} {record['score']:9.2f} {record['mu']:9.2f} {record['sigma']:9.2f} {record['matches']:8}")
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
