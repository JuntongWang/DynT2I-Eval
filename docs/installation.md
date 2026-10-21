# Installation

Use Linux, Python 3.10+, Conda for image generation workers, and NVIDIA GPUs with a matching PyTorch/CUDA installation. GPU memory requirements depend on the selected checkpoint, precision, image size, and evaluator. Generation adapters load a full pipeline onto one GPU; this release does not enable CPU offloading by default.

## CPU tools

```bash
python -m pip install -r requirements-core.txt
python scripts/demo_ranking.py --rounds 40 --seed 42
```

The demo, setup checker, ranking analysis, and tests do not require PyTorch or checkpoints.

## Scheduler environment

The scheduler itself uses the core requirements. It launches each generator through `conda run -n <environment> python ...`. The `model_envs` object in the runtime configuration selects the environment for each model; these names refer to environments you create, not environments included in the repository.

```bash
conda create -n dynt2i python=3.11 -y
conda activate dynt2i
python -m pip install -r requirements-core.txt
cp configs/runtime.example.json configs/runtime.json
```

## Generator environments

Install the publisher-recommended PyTorch, Diffusers, Transformers, and tokenizer dependencies in each generation environment. For a starting Diffusers environment:

```bash
conda create -n t2i-diffusers python=3.11 -y
conda activate t2i-diffusers
python -m pip install torch torchvision diffusers transformers accelerate safetensors sentencepiece protobuf Pillow
```

Set `model_envs.SDXL-Base-1.0` and `model_envs.SD-3.5-Medium` to `t2i-diffusers` to run the README example. Set their `model_paths` entries to downloaded checkpoint directories. Only selected models need to be configured.

Recent pipelines such as `Flux2KleinPipeline`, `LongCatImagePipeline`, and `ZImagePipeline` require Diffusers versions containing those classes. InternVL-U requires its separately installed source tree. Follow each publisher's installation instructions if a released package does not yet contain a required class.

## Prompt builder and alignment judge

Use an environment with `torch`, `torchvision`, `transformers`, `accelerate`, `qwen-vl-utils`, `fastapi`, and `uvicorn`. The quality prompt wrapper defaults to the Conda environment `qwen3.5`; override it with `--conda-env`.

Both builders and the alignment judge import `Qwen3_5ForConditionalGeneration`. Check support before downloading or loading the model:

```bash
python -c 'from transformers import Qwen3_5ForConditionalGeneration; print("Qwen3.5 support available")'
```

The Qwen3.5 model card bundled with the original experiment environment recommended current Transformers source. If your installed release lacks this class, use the publisher-recommended version. No full multi-model environment lockfile is supplied; preserve package versions and checkpoint revisions for runs you report.

## Perceptual and aesthetic judges

Use separate environments following the upstream instructions for VisualQuality-R1 and ArtiMuse. Configure their checkpoints and `external_code` paths. ArtiMuse's adapter uses its `InternVLChatModel` implementation and requests FlashAttention; install a compatible `flash-attn` build and upstream dependencies in that environment. The optional Q-Insight service also needs its source tree and a checkpoint with the `score_degradation` subfolder.

`requirements.txt` lists shared inference dependencies. It does not install external repositories, create Conda environments, select a CUDA build, or resolve checkpoint-specific version constraints.

## GPU assignment

The example reserves GPUs 0 and 1 for generation and GPU 2 for the evaluator. `CUDA_VISIBLE_DEVICES=2` makes physical GPU 2 appear as `cuda:0` inside that service, so leave that service's device setting as `cuda:0` (or `auto` for alignment). The two worker GPU IDs must differ. Run fewer dimensions concurrently when GPUs or memory are limited.

Evaluators receive absolute image paths; they must see the same filesystem as the scheduler. The default services listen on localhost. Wait for model loading to complete before starting a scheduler.
