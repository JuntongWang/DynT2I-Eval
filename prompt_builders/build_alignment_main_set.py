#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


from prompt_builders.common import ROOT, extract_json_from_output, latest_id, load_resource, resolve_builder_model_path


SYSTEM_PROMPT = """You are designing benchmark prompts for text-image alignment evaluation.
Fuse the provided tags into one realistic, visually coherent English image prompt and one atomic checklist.

Goals:
1. Preserve meaningful difficulty and discriminative power.
2. Make the scene physically plausible and visually judgeable.
3. Keep spatial relations, counting, and text rendering explicit when they appear.

Rules:
1. Use objective visual description only. No surrealism, impossible material conflicts, or decorative hype.
2. Rewrite the tags into natural prompt language while preserving their meaning.
3. Prefer 40 to 80 words.
4. The checklist must contain 4 to 6 yes/no questions, each testing one visible fact.
5. If a tag is about text rendering, keep the text content explicit in the prompt and checklist.
6. Do not use phrases like hyper-realistic, cinematic, volumetric, masterpiece.

Return strict JSON:
{"prompt":"...", "checklist":["...","...","...","..."]}"""


DIFFICULTY_TO_LOGIC_POOLS = {
    "easy": ["alignment_logic_binding", "alignment_logic_state"],
    "medium": ["alignment_logic_binding", "alignment_logic_counting", "alignment_logic_text_short"],
    "hard": ["alignment_logic_spatial", "alignment_logic_counting", "alignment_logic_text_short"],
    "hard_long_text": ["alignment_logic_text_long"],
}

DIFFICULTY_TEXTUAL_SUBJECT_PROB = {
    "easy": 0.02,
    "medium": 0.08,
    "hard": 0.10,
    "hard_long_text": 0.18,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-root", default=str(ROOT / "output" / "alignment_main"))
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def make_record_rng(seed: int | None, idx: int):
    if seed is None:
        return random.Random()
    return random.Random(seed * 1000003 + idx * 9176)


def sample_tags(pools, subject_pools, supports, difficulty: str, rng: random.Random):
    if rng.random() < DIFFICULTY_TEXTUAL_SUBJECT_PROB[difficulty]:
        subject = rng.choice(subject_pools["alignment_subject_textual_entity"])
    else:
        subject = rng.choice(subject_pools["alignment_subject_strong"])
    logic_pool = rng.choice(DIFFICULTY_TO_LOGIC_POOLS[difficulty])
    support_name = rng.choice(["environment", "lighting_atmosphere", "camera_composition", "medium_format"])
    support_label = support_name.replace("_", " ").title().replace(" And ", " & ")
    tags = {
        "Subject": subject,
        "Cognitive & Logic Constraints": rng.choice(pools[logic_pool]),
        support_label: rng.choice(supports[support_name]),
    }
    return tags, logic_pool


def generate_one(model, processor, tags: dict):
    import torch

    tag_text = "\n".join(f"- {key}: {value}" for key, value in tags.items())
    user_input = f"Fuse these tags into one benchmark prompt:\n{tag_text}"
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_input}]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=220)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return extract_json_from_output(output_text), output_text


def main():
    args = parse_args()
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    if args.count < 1:
        raise ValueError("Count must be positive.")
    model_path = resolve_builder_model_path(args.model_path)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    merged = output_root / "alignment_main.jsonl"
    manifest_path = output_root / "manifest.json"

    pools = load_resource("derived_prompt_pools.json")
    subject_pools = load_resource("subject_strength_pools.json")
    supports = {
        "environment": load_resource("environment.json"),
        "lighting_atmosphere": load_resource("lighting_atmosphere.json"),
        "camera_composition": load_resource("camera_composition.json"),
        "medium_format": load_resource("medium_format.json"),
    }

    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    start_id = latest_id(merged) if args.resume else 0
    mode = "a" if args.resume else "w"
    if start_id:
        print(f"Resuming from id {start_id + 1}", flush=True)

    run_counter = Counter()
    with open(merged, mode, encoding="utf-8") as handle:
        for offset in range(args.count):
            idx = start_id + offset + 1
            rng = make_record_rng(args.seed, idx)
            difficulty = rng.choice(["easy", "medium", "hard", "hard_long_text"])
            run_counter[difficulty] += 1
            tags, logic_pool = sample_tags(pools, subject_pools, supports, difficulty, rng)
            result, raw_output = generate_one(model, processor, tags)
            record = {
                "id": idx,
                "mode": "alignment_main",
                "difficulty": difficulty,
                "logic_pool": logic_pool,
                "source_tags": tags,
                "raw_output": raw_output,
            }
            if result:
                record["prompt"] = result.get("prompt", "")
                record["checklist"] = result.get("checklist", [])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"generated {offset + 1}/{args.count} -> id={idx} difficulty={difficulty} logic_pool={logic_pool}",
                flush=True,
            )

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}

    runs = manifest.get("runs", [])
    runs.append({"count": args.count, "seed": args.seed, "counts": dict(run_counter)})
    manifest = {"merged": str(merged), "num_runs": len(runs), "runs": runs}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"merged": str(merged), "counts": dict(run_counter)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
