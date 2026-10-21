from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.runtime import get_config_value, load_runtime_config, require_runtime_path


config = load_runtime_config()
source_root = require_runtime_path(
    get_config_value("external_code", "visualquality_r1_root", config=config),
    "external_code.visualquality_r1_root",
)
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from qwen_vl_utils import process_vision_info


app = FastAPI(title="Perceptual Quality Server")
model = None
processor = None
device = get_config_value("devices", "perceptual_quality", default="cuda:0", config=config)
model_path = require_runtime_path(
    get_config_value("model_paths", "VisualQuality-R1-7B", config=config),
    "model_paths.VisualQuality-R1-7B",
)


class IQARequest(BaseModel):
    image_path: str
    k_samples: int = 4
    temperature: float = 0.7
    return_reasonings: bool = False


def extract_reasoning_and_score(output_text: str) -> tuple[str, float]:
    reasoning_matches = re.findall(r"<think>(.*?)</think>", output_text, re.DOTALL)
    reasoning = reasoning_matches[-1].strip() if reasoning_matches else "No reasoning found."

    try:
        answer_matches = re.findall(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
        answer_text = answer_matches[-1].strip() if answer_matches else output_text.strip()
        score = float(re.search(r"\d+(?:\.\d+)?", answer_text).group())
        score = max(1.0, min(5.0, score))
    except Exception:
        print(f"[PerceptualQualityServer] Failed to parse score. Fallback to 3.0. Raw output: {output_text}")
        score = 3.0
    return reasoning, score


@app.on_event("startup")
def load_model() -> None:
    global model, processor
    print(f"[PerceptualQualityServer] Loading model to {device}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    print("[PerceptualQualityServer] Model is ready.")


@app.post("/evaluate_iqa")
async def evaluate_iqa(request: IQARequest):
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")

    k_samples = max(1, int(request.k_samples))
    prompt = (
        "You are an expert evaluator for AI-generated images, focusing strictly on perceptual and technical image quality. "
        "Rate this image on a scale of 1 to 5 based only on visible generation quality. "
        "Do not judge creativity, artistic taste, emotional impact, or aesthetic style unless they directly reflect generation quality. "
        "Focus on structural integrity, realism of textures and materials, local detail consistency, lighting coherence, anatomy, geometry, spatial consistency, textural oversmoothing, noise, pixelation, and object-boundary blending errors. "
        "Do not reward saturated color, dramatic contrast, or fashionable style by themselves. "
        "Output a float rounded to two decimal places."
    )
    if request.return_reasonings:
        question_text = (
            f"{prompt} First output the thinking process in <think> </think> tags "
            "and then output the final answer with only one score in <answer> </answer> tags."
        )
    else:
        question_text = f"{prompt} Output only one score in <answer> </answer> tags."

    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": request.image_path},
                {"type": "text", "text": question_text},
            ],
        }
    ]

    text = [
        processor.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=True,
        )
        for msg in [message]
    ]
    image_inputs, video_inputs = process_vision_info([message])
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    raw_scores = []
    reasonings = []
    for _ in range(k_samples):
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=2048,
                do_sample=True,
                temperature=request.temperature,
                top_k=50,
                top_p=1,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        reasoning, score = extract_reasoning_and_score(output_text)
        raw_scores.append(score)
        if request.return_reasonings:
            reasonings.append(reasoning)

    mean_score = float(np.mean(raw_scores))
    variance = float(np.var(raw_scores, ddof=1)) if len(raw_scores) > 1 else 0.0
    std = float(np.sqrt(variance))
    return {
        "score": mean_score,
        "variance": variance,
        "std": std,
        "raw_scores": raw_scores,
        "reasonings": reasonings,
        "num_samples": k_samples,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
