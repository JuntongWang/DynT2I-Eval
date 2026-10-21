from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration, set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.runtime import get_config_value, load_runtime_config, require_runtime_path


config = load_runtime_config()
source_root = require_runtime_path(
    get_config_value("external_code", "q_insight_root", config=config),
    "external_code.q_insight_root",
)
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from qwen_vl_utils import process_vision_info


app = FastAPI(title="Q-Insight IQA Server")
model = None
processor = None
device = get_config_value("devices", "q_insight", default="cuda:0", config=config)
model_path = require_runtime_path(
    get_config_value("model_paths", "Q-Insight", config=config),
    "model_paths.Q-Insight",
)


class IQARequest(BaseModel):
    image_path: str


def extract_score(output_text: str) -> float:
    answer_match = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
    text_to_parse = answer_match.group(1) if answer_match else output_text

    try:
        json_match = re.search(r"\{[^{}]*rating[^{}]*\}", text_to_parse, re.IGNORECASE)
        if json_match:
            data = json.loads(json_match.group(0))
            return float(data.get("rating", data.get("Rating", 0)))
    except json.JSONDecodeError:
        pass

    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text_to_parse)
    if numbers:
        return float(numbers[-1])
    return -1.0


@app.on_event("startup")
def load_model() -> None:
    global model, processor
    print(f"[QInsightServer] Loading model to {device}...")
    set_seed(42)
    subfolder = "score_degradation"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        subfolder=subfolder,
    )
    processor = AutoProcessor.from_pretrained(model_path, subfolder=subfolder)
    print("[QInsightServer] Model is ready.")


@app.post("/evaluate_iqa")
async def evaluate_iqa(request: IQARequest):
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")

    system_prompt = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags."
    )
    score_prompt = (
        "First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. "
        "What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, "
        "rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. "
        "Return the final answer in JSON format with the key \"rating\"."
    )
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        max_new_tokens=1024,
    )

    message = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": score_prompt},
                {"type": "image", "image": f"file://{request.image_path}"},
            ],
        },
    ]

    text = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
    image_inputs, video_inputs = process_vision_info([message])
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            generation_config=generation_config,
            use_cache=True,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return {"score": extract_score(output_text)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
