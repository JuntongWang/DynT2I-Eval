# Models and evaluators

`benchmark/model_registry.py` defines generation parameters, pipeline names, and default Conda environments. `configs/runtime.json` supplies checkpoint locations and environment overrides. Registry names are exact and case sensitive.

| Registry name | Adapter / pipeline | Default environment |
| --- | --- | --- |
| `CogView4-6B` | `CogView4Pipeline` | `cogview` |
| `FLUX.1-dev` | `FluxPipeline` | `flux` |
| `FLUX.1-Krea-dev` | `FluxPipeline` | `flux` |
| `FLUX.2-klein-4B` | `Flux2KleinPipeline` | `flux2-klein` |
| `FLUX.2-klein-9B` | `Flux2KleinPipeline` | `flux2-klein` |
| `LongCat-Image` | `LongCatImagePipeline` | `longcat-image` |
| `SD-3-Medium` | `StableDiffusion3Pipeline` | `sd3.5` |
| `SD-3.5-Large` | `StableDiffusion3Pipeline` | `sd3.5` |
| `SD-3.5-Medium` | `StableDiffusion3Pipeline` | `sd3.5` |
| `SD3.5-Large-Turbo` | `StableDiffusion3Pipeline` | `sd3.5` |
| `SDXL-Base-1.0` | `DiffusionPipeline` | `sd3.5` |
| `Z-Image-Turbo` | `ZImagePipeline` | `z-image` |
| `InternVL-U` | External `InternVLUPipeline` | `internvl-u` |

The default order is the registry's insertion order. Use `--models` to select a subset. Generation parameters are recorded in `ModelSpec.generate_kwargs`; SDXL leaves generation settings at the pipeline defaults. Preserve the registry and upstream package versions with reported results.

## Add a generation model

1. Add a `ModelSpec` in `benchmark/model_registry.py` with the pipeline class and inference settings. For a specialized implementation, subclass `BaseImageGenerator` instead of using the generic Diffusers adapter.
2. Add the adapter mapping in `benchmark/generators/factory.py`. Existing thin adapters show how to wrap `RegisteredDiffusersGenerator`.
3. Add checkpoint and environment entries in your local configuration.
4. Include the registry name in `--models` when starting or resuming a run. It receives a fresh prior if absent from saved state.

## Evaluators

| Model | Configuration | Default endpoint |
| --- | --- | --- |
| Qwen3.5-9B | `model_paths.Qwen3.5-9B`, `devices.alignment_judge` | `http://127.0.0.1:8000/evaluate` |
| VisualQuality-R1-7B | `model_paths.VisualQuality-R1-7B`, `external_code.visualquality_r1_root` | `http://127.0.0.1:8001/evaluate_iqa` |
| ArtiMuse | `model_paths.ArtiMuse`, `external_code.artimuse_root` | `http://127.0.0.1:8002/evaluate_aesthetic` |
| Q-Insight (optional) | `model_paths.Q-Insight`, `external_code.q_insight_root` | `http://127.0.0.1:8001/evaluate_iqa` |

Start Q-Insight with `python -m benchmark.evaluator.q_insight_server` in place of the default perceptual service. It returns one score per image rather than a sampled variance estimate, so it changes the evaluation protocol and should be identified in reported results.

All evaluators access images by local path. Obtain checkpoints and external implementations under their publishers' terms; none are included in this code release.
