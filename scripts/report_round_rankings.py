#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_ROUNDS = [200, 210, 220, 230, 240, 250]


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Read score_history.jsonl snapshots and print rankings for selected rounds."
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
        "--rounds",
        nargs="+",
        type=int,
        default=DEFAULT_ROUNDS,
        help="Target rounds to print.",
    )
    return parser.parse_args()


def load_round_snapshots(score_history_path: Path, target_rounds: set[int]) -> dict[int, dict]:
    snapshots: dict[int, dict] = {}
    with score_history_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            round_id = payload.get("round")
            if round_id in target_rounds:
                snapshots[round_id] = payload
    return snapshots


def sort_leaderboard(leaderboard: dict[str, dict]) -> list[tuple[str, dict]]:
    if all("rank" in item for item in leaderboard.values()):
        return sorted(leaderboard.items(), key=lambda pair: (pair[1]["rank"], -pair[1]["score"], pair[0]))
    return sorted(leaderboard.items(), key=lambda pair: (-pair[1]["score"], pair[0]))


def format_snapshot(output_dir: Path, round_id: int, snapshot: dict | None) -> str:
    lines = [f"## {output_dir.name} | Round {round_id}"]
    if snapshot is None:
        lines.append("MISSING")
        return "\n".join(lines)

    matchup = snapshot.get("matchup", {})
    lines.append(
        "matchup: "
        f"{matchup.get('model_a', 'NA')} vs {matchup.get('model_b', 'NA')}, "
        f"winner={matchup.get('winner', 'NA')}, "
        f"wins_a={matchup.get('wins_a', 'NA')}, "
        f"wins_b={matchup.get('wins_b', 'NA')}, "
        f"ties={matchup.get('ties', 'NA')}, "
        f"batch_weight={matchup.get('batch_weight', 'NA')}"
    )

    for model_name, stats in sort_leaderboard(snapshot.get("leaderboard", {})):
        lines.append(
            f"{stats['rank']:>2}. {model_name} | "
            f"score={stats['score']:.2f} | "
            f"mu={stats['mu']:.2f} | "
            f"sigma={stats['sigma']:.2f} | "
            f"matches={stats['matches']} | "
            f"win_rate={stats['win_rate']:.3f} | "
            f"status={stats['status']}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    target_rounds = set(args.rounds)

    blocks: list[str] = []
    for output_dir_str in args.output_dirs:
        output_dir = Path(output_dir_str).resolve()
        score_history_path = output_dir / "score_history.jsonl"
        if not score_history_path.exists():
            blocks.append(f"## {output_dir.name}\nMISSING score history: {score_history_path}")
            continue

        snapshots = load_round_snapshots(score_history_path, target_rounds)
        for round_id in args.rounds:
            blocks.append(format_snapshot(output_dir, round_id, snapshots.get(round_id)))

    print("\n\n".join(blocks))


if __name__ == "__main__":
    main()
