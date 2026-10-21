#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


from prompt_builders.common import extract_json_from_output, latest_id, load_resource, resolve_builder_model_path


SYSTEM_PROMPT = """You are designing benchmark prompts for perceptual quality and aesthetic quality evaluation.
Fuse the provided tags into one realistic English image prompt.

Goals:
1. The scene should be natural, readable, and visually rich.
2. Semantic burden must remain low enough that IQA and aesthetic judgment are not dominated by instruction-following failure.

Rules:
1. Keep the scene physically plausible.
2. Focus on concrete materials, lighting, framing, and scene clarity.
3. Use at most one mild visual logic constraint.
4. Prefer 22 to 48 words.
5. Use simple, clean English. Avoid poetic or cinematic filler.
6. Do not use stock style phrases such as hyper-realistic, cinematic, volumetric, masterpiece, or award-winning.
7. Return strict JSON only.

Return strict JSON:
{"prompt":"..."}"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--logic-prob", type=float, default=0.03)
    parser.add_argument("--medium-prob", type=float, default=0.12)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def make_record_rng(seed: int | None, idx: int, attempt: int = 0):
    if seed is None:
        return random.Random()
    return random.Random(seed * 1000003 + idx * 9176 + attempt * 37)


def prompt_looks_broken(prompt: str) -> bool:
    if not prompt.strip():
        return True
    if len(prompt.split()) < 10:
        return True
    if re.search(r"[\ufffd]", prompt):
        return True
    if re.search(r"\*\*", prompt):
        return True
    if re.search(r"[\u4e00-\u9fff].*[A-Za-z].*[\u4e00-\u9fff]", prompt):
        return True
    return False


def sample_tags(pools, subject_pools, supports, logic_prob: float, medium_prob: float, rng: random.Random):
    tags = {"Subject": rng.choice(subject_pools["quality_subject_strong"])}
    core_supports = ["environment", "lighting_atmosphere", "camera_composition"]
    chosen = rng.sample(core_supports, k=2)
    if rng.random() < medium_prob:
        chosen.append("medium_format")
    for support_name in chosen:
        support_label = support_name.replace("_", " ").title().replace(" And ", " & ")
        tags[support_label] = rng.choice(supports[support_name])
    if rng.random() < logic_prob:
        tags["Cognitive & Logic Constraints"] = rng.choice(pools["quality_logic_soft"])
    return tags


def generate_one(model, processor, tags: dict):
    import torch

    tag_text = "\n".join(f"- {key}: {value}" for key, value in tags.items())
    user_input = (
        "Write one short realistic image prompt as strict JSON.\n"
        "Keep it visually clear and physically plausible.\n"
        f"{tag_text}"
    )
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
        generated_ids = model.generate(**inputs, max_new_tokens=120)
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

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_id = latest_id(out_path) if args.resume else 0
    mode = "a" if args.resume else "w"
    if start_id:
        print(f"Resuming from id {start_id + 1}", flush=True)

    with open(out_path, mode, encoding="utf-8") as handle:
        for offset in range(args.count):
            idx = start_id + offset + 1
            final_tags = None
            result = None
            raw_output = ""
            last_tags = None
            for attempt in range(args.max_attempts):
                rng = make_record_rng(args.seed, idx, attempt)
                logic_prob = args.logic_prob if attempt == 0 else 0.0
                medium_prob = args.medium_prob if attempt == 0 else 0.0
                tags = sample_tags(pools, subject_pools, supports, logic_prob, medium_prob, rng)
                last_tags = tags
                result, raw_output = generate_one(model, processor, tags)
                prompt = result.get("prompt", "") if result else ""
                if result and not prompt_looks_broken(prompt):
                    final_tags = tags
                    break

            record = {
                "id": idx,
                "mode": "quality",
                "source_tags": final_tags or last_tags or {},
                "raw_output": raw_output,
            }
            if result:
                record["prompt"] = result.get("prompt", "")
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"generated {offset + 1}/{args.count} -> id={idx}", flush=True)


if __name__ == "__main__":
    main()
