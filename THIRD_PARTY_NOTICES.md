# Third-party resources

The MIT license covers project-authored code. It does not replace the licenses of model weights, datasets, or third-party source code used at runtime.

## Prompt resource pools

The six JSON files in `prompt_builders/resources/` contain visual semantic phrases and task-specific pools extracted and reorganized from descriptions in **DOCCI (Descriptions of Connected and Contrasting Images)**. The paper describes this source in Section 3.1. The original DOCCI descriptions and images are not bundled as a dataset here.

- Dataset and source terms: <https://google.github.io/docci/>
- Source dataset: <https://huggingface.co/datasets/google/docci>

These files are derived data. Their use and redistribution remain subject to applicable source dataset terms. Extraction and pool construction scripts are not included in this release; the supplied pools are the starting point for prompt generation.

## Runtime dependencies

These projects are referenced through installed packages, local checkpoints, or separately obtained source directories:

| Project | Use in DynT2I-Eval |
| --- | --- |
| Qwen3.5 | Prompt rewriting and image-order-swapped alignment judgments |
| VisualQuality-R1 | Perceptual quality scores and sample variance |
| ArtiMuse | Aesthetic quality scores |
| Q-Insight | Optional alternative quality evaluator |
| Hugging Face Diffusers / Transformers | Loading and inference for supported checkpoints |
| InternVL-U | Generation adapter requiring its external implementation |

See [model configuration](docs/models.md) for the remaining generation models. Obtain models from their official publishers and follow checkpoint-specific licenses and installation instructions. The repository does not bundle those weights or external source trees.
