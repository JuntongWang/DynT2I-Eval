#!/usr/bin/env python3
import argparse
import json
from itertools import combinations
from pathlib import Path


DEFAULT_CHECKPOINTS = [200, 210, 220, 230, 240, 250]


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Analyze ranking stability from score_history.jsonl snapshots."
    )
    parser.add_argument(
        "--output-dirs",
        nargs="+",
        default=[
            str(base_dir / "benchmark_outputs" / "text_alignment"),
            str(base_dir / "benchmark_outputs" / "perceptual_quality"),
            str(base_dir / "benchmark_outputs" / "aesthetic_quality"),
        ],
        help="Benchmark output directories that contain score_history.jsonl.",
    )
    parser.add_argument(
        "--reference-round",
        type=int,
        default=200,
        help="Models activated after this round are treated as newly added.",
    )
    parser.add_argument(
        "--end-round",
        type=int,
        default=250,
        help="Main analysis horizon.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=DEFAULT_CHECKPOINTS,
        help="Checkpoint rounds used for interval comparisons.",
    )
    return parser.parse_args()


def load_history(score_history_path: Path) -> dict[int, dict]:
    history = {}
    with score_history_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            history[payload["round"]] = payload
    return history


def first_active_rounds(history: dict[int, dict]) -> dict[str, int]:
    first_round = {}
    for round_id in sorted(history):
        for model_name, stats in history[round_id]["leaderboard"].items():
            if stats["matches"] > 0 and model_name not in first_round:
                first_round[model_name] = round_id
    return first_round


def subset_order(leaderboard: dict[str, dict], subset: list[str] | None) -> list[str]:
    models = list(leaderboard.keys()) if subset is None else [model for model in subset if model in leaderboard]
    return sorted(models, key=lambda model: leaderboard[model]["rank"])


def subset_ranks(leaderboard: dict[str, dict], subset: list[str] | None) -> dict[str, int]:
    order = subset_order(leaderboard, subset)
    return {model: index for index, model in enumerate(order, start=1)}


def kendall_tau(rank_a: dict[str, int], rank_b: dict[str, int]) -> float:
    items = list(rank_a.keys())
    if len(items) < 2:
        return 1.0

    concordant = 0
    discordant = 0
    for left, right in combinations(items, 2):
        sign_a = rank_a[left] - rank_a[right]
        sign_b = rank_b[left] - rank_b[right]
        if sign_a * sign_b > 0:
            concordant += 1
        elif sign_a * sign_b < 0:
            discordant += 1

    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def interval_metrics(history: dict[int, dict], round_a: int, round_b: int, subset: list[str] | None) -> dict | None:
    if round_a not in history or round_b not in history:
        return None

    ranks_a = subset_ranks(history[round_a]["leaderboard"], subset)
    ranks_b = subset_ranks(history[round_b]["leaderboard"], subset)
    common = sorted(set(ranks_a) & set(ranks_b))
    if not common:
        return None

    ranks_a = {model: ranks_a[model] for model in common}
    ranks_b = {model: ranks_b[model] for model in common}
    order_a = sorted(common, key=lambda model: ranks_a[model])
    order_b = sorted(common, key=lambda model: ranks_b[model])

    top3_a = set(order_a[:3])
    top3_b = set(order_b[:3])
    top5_a = set(order_a[:5])
    top5_b = set(order_b[:5])

    return {
        "tau": kendall_tau(ranks_a, ranks_b),
        "avg_shift": sum(abs(ranks_a[model] - ranks_b[model]) for model in common) / len(common),
        "top3_overlap": len(top3_a & top3_b) / min(3, len(common)),
        "top5_overlap": len(top5_a & top5_b) / min(5, len(common)),
    }


def stable_from_round(history: dict[int, dict], start_round: int, end_round: int, subset: list[str] | None) -> int | None:
    rounds = [round_id for round_id in sorted(history) if start_round <= round_id <= end_round]
    if not rounds:
        return None

    orders = {round_id: tuple(subset_order(history[round_id]["leaderboard"], subset)) for round_id in rounds}
    for index, round_id in enumerate(rounds):
        current = orders[round_id]
        if all(orders[future_round] == current for future_round in rounds[index:]):
            return round_id
    return None


def format_float(value: float) -> str:
    return f"{value:.3f}"


def analyze_directory(output_dir: Path, reference_round: int, end_round: int, checkpoints: list[int]) -> str:
    score_history_path = output_dir / "score_history.jsonl"
    if not score_history_path.exists():
        return f"## {output_dir.name}\nMISSING score history: {score_history_path}"

    history = load_history(score_history_path)
    first_round = first_active_rounds(history)

    legacy_models = sorted(
        model_name for model_name, round_id in first_round.items() if round_id <= reference_round
    )
    new_models = sorted(
        model_name for model_name, round_id in first_round.items() if reference_round < round_id <= end_round
    )
    late_models = sorted(
        model_name for model_name, round_id in first_round.items() if round_id > end_round
    )

    lines = [f"## {output_dir.name}"]
    lines.append(f"analysis_window: rounds {reference_round} -> {end_round}")
    lines.append(
        "legacy_models: "
        + ", ".join(f"{model}({first_round[model]})" for model in legacy_models)
    )
    lines.append(
        "new_models_in_window: "
        + (", ".join(f"{model}({first_round[model]})" for model in new_models) if new_models else "none")
    )
    if late_models:
        lines.append(
            "late_models_after_window: "
            + ", ".join(f"{model}({first_round[model]})" for model in late_models)
        )

    for subset_name, subset in [("all", None), ("legacy", legacy_models)]:
        stable_round = stable_from_round(history, reference_round, end_round, subset)
        lines.append(f"stable_from_{subset_name}: {stable_round}")

    for subset_name, subset in [("all", None), ("legacy", legacy_models)]:
        metrics = interval_metrics(history, reference_round, end_round, subset)
        if metrics is None:
            continue
        lines.append(
            f"{subset_name}_{reference_round}_to_{end_round}: "
            f"tau={format_float(metrics['tau'])}, "
            f"avg_shift={format_float(metrics['avg_shift'])}, "
            f"top3_overlap={format_float(metrics['top3_overlap'])}, "
            f"top5_overlap={format_float(metrics['top5_overlap'])}"
        )

    lines.append("checkpoint_intervals:")
    for subset_name, subset in [("all", None), ("legacy", legacy_models)]:
        lines.append(f"subset={subset_name}")
        for round_a, round_b in zip(checkpoints, checkpoints[1:]):
            metrics = interval_metrics(history, round_a, round_b, subset)
            if metrics is None:
                lines.append(f"{round_a}->{round_b}: MISSING")
                continue
            lines.append(
                f"{round_a}->{round_b}: "
                f"tau={format_float(metrics['tau'])}, "
                f"avg_shift={format_float(metrics['avg_shift'])}, "
                f"top3_overlap={format_float(metrics['top3_overlap'])}, "
                f"top5_overlap={format_float(metrics['top5_overlap'])}"
            )

    if new_models:
        lines.append("new_model_trajectories:")
        for model_name in new_models:
            parts = [f"{model_name}"]
            for round_id in checkpoints:
                snapshot = history.get(round_id)
                if snapshot is None or model_name not in snapshot["leaderboard"]:
                    parts.append(f"{round_id}:NA")
                    continue
                stats = snapshot["leaderboard"][model_name]
                parts.append(
                    f"{round_id}:rank={stats['rank']},score={stats['score']:.2f},"
                    f"sigma={stats['sigma']:.2f},matches={stats['matches']}"
                )
            lines.append(" | ".join(parts))

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checkpoints = [round_id for round_id in args.checkpoints if args.reference_round <= round_id <= args.end_round]
    blocks = []
    for output_dir_str in args.output_dirs:
        output_dir = Path(output_dir_str).resolve()
        blocks.append(
            analyze_directory(
                output_dir=output_dir,
                reference_round=args.reference_round,
                end_round=args.end_round,
                checkpoints=checkpoints,
            )
        )
    print("\n\n".join(blocks))


if __name__ == "__main__":
    main()
