from __future__ import annotations

import sys

import torch

from benchmark.generators.base import BaseImageGenerator
from benchmark.runtime import get_config_value, load_runtime_config, require_runtime_path


class InternVLUGenerator(BaseImageGenerator):
    def __init__(
        self,
        base_output_dir: str,
        model_path: str | None = None,
        source_root: str | None = None,
        config_path: str | None = None,
    ):
        super().__init__(model_name="InternVL-U", base_output_dir=base_output_dir)
        config = load_runtime_config(config_path)
        self.model_path = require_runtime_path(
            model_path or get_config_value("model_paths", "InternVL-U", config=config),
            "model_paths.InternVL-U",
        )
        self.source_root = require_runtime_path(
            source_root or get_config_value("external_code", "internvl_u_root", config=config),
            "external_code.internvl_u_root",
        )

    def load_model(self) -> None:
        if not torch.cuda.is_available():
            raise SystemExit("InternVL-U requires CUDA.")
        if self.source_root not in sys.path:
            sys.path.insert(0, self.source_root)
        from internvlu import InternVLUPipeline

        self.model_pipeline = InternVLUPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        )
        self.model_pipeline = self.model_pipeline.to("cuda")

    def generate_single_image(self, prompt: str, seed: int):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        return self.model_pipeline(
            prompt=prompt,
            generation_mode="image",
            height=1024,
            width=1024,
            generator=generator,
        ).images[0]
