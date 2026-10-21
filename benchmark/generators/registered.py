from __future__ import annotations

import importlib

import torch

from benchmark.generators.base import BaseImageGenerator
from benchmark.model_registry import MODEL_SPECS
from benchmark.runtime import get_config_value, load_runtime_config, require_runtime_path


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


class RegisteredDiffusersGenerator(BaseImageGenerator):
    def __init__(self, model_name: str, base_output_dir: str, config_path: str | None = None):
        super().__init__(model_name=model_name, base_output_dir=base_output_dir)
        self.config_path = config_path
        self.config = load_runtime_config(config_path)
        self.spec = MODEL_SPECS[model_name]
        self.model_path = require_runtime_path(
            get_config_value("model_paths", model_name, config=self.config),
            f"model_paths.{model_name}",
        )

    def load_model(self) -> None:
        if not torch.cuda.is_available():
            raise SystemExit(f"{self.model_name} requires CUDA.")
        diffusers_module = importlib.import_module("diffusers")
        pipeline_cls = getattr(diffusers_module, self.spec.pipeline_class)
        self.model_pipeline = pipeline_cls.from_pretrained(
            self.model_path,
            torch_dtype=resolve_torch_dtype(self.spec.torch_dtype),
            **self.spec.load_kwargs,
        )
        self.model_pipeline = self.model_pipeline.to("cuda")

    def generate_single_image(self, prompt: str, seed: int):
        generator = torch.Generator(self.spec.generator_device).manual_seed(seed)
        generation_kwargs = dict(self.spec.generate_kwargs)
        generation_kwargs["prompt"] = prompt
        generation_kwargs["generator"] = generator
        return self.model_pipeline(**generation_kwargs).images[0]
