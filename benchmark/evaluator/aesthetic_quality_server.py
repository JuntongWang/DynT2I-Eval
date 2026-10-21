from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.runtime import get_config_value, load_runtime_config, require_runtime_path


config = load_runtime_config()
source_root = require_runtime_path(
    get_config_value("external_code", "artimuse_root", config=config),
    "external_code.artimuse_root",
)

source_root_path = Path(source_root)
if source_root not in sys.path:
    sys.path.insert(0, source_root)
src_dir = source_root_path / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
artimuse_inner = src_dir / "artimuse"
if artimuse_inner.exists() and str(artimuse_inner) not in sys.path:
    sys.path.insert(0, str(artimuse_inner))

from artimuse.internvl.model.internvl_chat.modeling_artimuse import InternVLChatModel


app = FastAPI(title="Aesthetic Quality Server")
model = None
tokenizer = None
device = get_config_value("devices", "aesthetic_quality", default="cuda:0", config=config)
model_path = require_runtime_path(
    get_config_value("model_paths", "ArtiMuse", config=config),
    "model_paths.ArtiMuse",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AestheticRequest(BaseModel):
    image_path: str


def build_transform(input_size: int = 448):
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_image(image_file: str, input_size: int = 448):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size)
    return transform(image).unsqueeze(0)


@app.on_event("startup")
def load_model() -> None:
    global model, tokenizer
    print(f"[AestheticQualityServer] Loading model to {device}...")
    model = InternVLChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    print("[AestheticQualityServer] Model is ready.")


@app.post("/evaluate_aesthetic")
async def evaluate_aesthetic(request: AestheticRequest):
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")

    try:
        generation_config = dict(
            max_new_tokens=8192,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        pixel_values = load_image(request.image_path).to(torch.bfloat16).to(device)
        with torch.no_grad():
            score = float(model.score(device, tokenizer, pixel_values, generation_config))
        return {
            "score": score,
            "variance": 0.0,
            "std": 0.0,
            "raw_scores": [score],
            "num_samples": 1,
            "num_views": 1,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
