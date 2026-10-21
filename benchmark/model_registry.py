from __future__ import annotations

from dataclasses import dataclass, field

from benchmark.runtime import get_config_value


@dataclass(frozen=True)
class ModelSpec:
    conda_env: str
    pipeline_class: str | None = None
    torch_dtype: str = "bfloat16"
    load_kwargs: dict = field(default_factory=dict)
    generate_kwargs: dict = field(default_factory=dict)
    generator_device: str = "cuda"
    kind: str = "diffusers"


MODEL_SPECS: dict[str, ModelSpec] = {
    "CogView4-6B": ModelSpec(
        conda_env="cogview",
        pipeline_class="CogView4Pipeline",
        generate_kwargs={
            "guidance_scale": 3.5,
            "num_inference_steps": 50,
            "width": 1024,
            "height": 1024,
            "num_images_per_prompt": 1,
        },
    ),
    "FLUX.1-dev": ModelSpec(
        conda_env="flux",
        pipeline_class="FluxPipeline",
        generate_kwargs={
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.5,
            "num_inference_steps": 50,
            "max_sequence_length": 512,
        },
    ),
    "FLUX.1-Krea-dev": ModelSpec(
        conda_env="flux",
        pipeline_class="FluxPipeline",
        generate_kwargs={
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.5,
            "num_inference_steps": 50,
            "max_sequence_length": 512,
        },
    ),
    "FLUX.2-klein-4B": ModelSpec(
        conda_env="flux2-klein",
        pipeline_class="Flux2KleinPipeline",
        generate_kwargs={
            "height": 1024,
            "width": 1024,
            "guidance_scale": 1.0,
            "num_inference_steps": 4,
        },
    ),
    "FLUX.2-klein-9B": ModelSpec(
        conda_env="flux2-klein",
        pipeline_class="Flux2KleinPipeline",
        generate_kwargs={
            "height": 1024,
            "width": 1024,
            "guidance_scale": 1.0,
            "num_inference_steps": 4,
        },
    ),
    "LongCat-Image": ModelSpec(
        conda_env="longcat-image",
        pipeline_class="LongCatImagePipeline",
        generator_device="cpu",
        generate_kwargs={
            "height": 768,
            "width": 1344,
            "guidance_scale": 4.0,
            "num_inference_steps": 50,
            "enable_cfg_renorm": True,
            "enable_prompt_rewrite": True,
        },
    ),
    "SD-3.5-Large": ModelSpec(
        conda_env="sd3.5",
        pipeline_class="StableDiffusion3Pipeline",
        generate_kwargs={"num_inference_steps": 28, "guidance_scale": 3.5},
    ),
    "SD-3.5-Medium": ModelSpec(
        conda_env="sd3.5",
        pipeline_class="StableDiffusion3Pipeline",
        generate_kwargs={"num_inference_steps": 40, "guidance_scale": 4.5},
    ),
    "SD3.5-Large-Turbo": ModelSpec(
        conda_env="sd3.5",
        pipeline_class="StableDiffusion3Pipeline",
        generate_kwargs={"num_inference_steps": 4, "guidance_scale": 0.0},
    ),
    "SD-3-Medium": ModelSpec(
        conda_env="sd3.5",
        pipeline_class="StableDiffusion3Pipeline",
        torch_dtype="float16",
        generate_kwargs={"negative_prompt": "", "num_inference_steps": 28, "guidance_scale": 7.0},
    ),
    "SDXL-Base-1.0": ModelSpec(
        conda_env="sd3.5",
        pipeline_class="DiffusionPipeline",
        torch_dtype="float16",
        load_kwargs={"use_safetensors": True, "variant": "fp16"},
        generate_kwargs={},
    ),
    "Z-Image-Turbo": ModelSpec(
        conda_env="z-image",
        pipeline_class="ZImagePipeline",
        load_kwargs={"low_cpu_mem_usage": False},
        generate_kwargs={
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
        },
    ),
    "InternVL-U": ModelSpec(conda_env="internvl-u", kind="internvl_u"),
}


DEFAULT_MODEL_ORDER = list(MODEL_SPECS.keys())


def get_model_env(model_name: str, config: dict | None = None) -> str:
    override = get_config_value("model_envs", model_name, config=config)
    if override:
        return str(override)
    return MODEL_SPECS[model_name].conda_env
