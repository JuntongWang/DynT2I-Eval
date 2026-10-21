from __future__ import annotations

import importlib

from benchmark.generators.base import BaseImageGenerator
MODEL_CLASS_MAP = {
    "CogView4-6B": ("benchmark.generators.cogview4", "CogView4Generator"),
    "FLUX.1-dev": ("benchmark.generators.flux_dev", "FluxGenerator"),
    "FLUX.1-Krea-dev": ("benchmark.generators.flux_krea", "FluxKreaGenerator"),
    "FLUX.2-klein-4B": ("benchmark.generators.flux_2_klein_4b", "Flux2Klein4BGenerator"),
    "FLUX.2-klein-9B": ("benchmark.generators.flux_2_klein_9b", "Flux2Klein9BGenerator"),
    "LongCat-Image": ("benchmark.generators.longcat", "LongCatGenerator"),
    "SD-3.5-Large": ("benchmark.generators.sd3_5", "SD35Generator"),
    "SD-3.5-Medium": ("benchmark.generators.sd3_5_medium", "SD35MediumGenerator"),
    "SD3.5-Large-Turbo": ("benchmark.generators.sd3_5_turbo", "SD35TurboGenerator"),
    "SD-3-Medium": ("benchmark.generators.sd3_medium", "SD3MediumGenerator"),
    "SDXL-Base-1.0": ("benchmark.generators.sdxl_base", "SDXLBaseGenerator"),
    "Z-Image-Turbo": ("benchmark.generators.z_image_turbo", "ZImageGenerator"),
    "InternVL-U": ("benchmark.generators.internvl_u", "InternVLUGenerator"),
}


def create_generator(model_name: str, output_dir: str, config_path: str | None = None) -> BaseImageGenerator:
    module_name, class_name = MODEL_CLASS_MAP[model_name]
    module = importlib.import_module(module_name)
    generator_cls = getattr(module, class_name)
    return generator_cls(base_output_dir=output_dir, config_path=config_path)
