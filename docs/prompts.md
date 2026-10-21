# Prompt streams

The supplied semantic pools were extracted and reorganized from DOCCI descriptions. See [third-party notices](../THIRD_PARTY_NOTICES.md). Pool extraction is not part of this release; prompt builders start from the JSON resources.

## Text alignment

```bash
python build_text_alignment_set.py --count 200 --seed 42
```

Each record samples a strong or textual subject, a logic constraint, and one supporting dimension (environment, lighting, composition, or medium). A Qwen3.5 model rewrites these tags into an image prompt and an atomic checklist.

| Difficulty | Logic pools |
| --- | --- |
| `easy` | Attribute binding, state/action |
| `medium` | Attribute binding, counting, short text |
| `hard` | Spatial relations, counting, short text |
| `hard_long_text` | Long text rendering |

The builder samples the four tiers with equal probability; a finite stream need not have exactly equal counts. Defaults write `prompt_builders/output/alignment_main/alignment_main.jsonl` and a run manifest. Use `--output-root` for another directory or `--no-resume` to replace an existing stream.

## Perceptual and aesthetic quality

```bash
python build_perceptual_quality_set.py --count 200 --seed 42 --gpu 2 --conda-env qwen3.5
```

The quality stream samples a strong subject and two supporting dimensions, with optional medium information and a low probability of a mild logic constraint. Perceptual and aesthetic schedulers use the same prompt format. Defaults write `prompt_builders/output/quality_main/quality_main.jsonl`, a manifest, and an analysis report.

The wrapper appends by default. For lower-level sampling and retry controls, run:

```bash
python -m prompt_builders.generate_quality_vnext --help
```

## Data format

Each line is a JSON object with a unique `id` and a nonempty `prompt`. Alignment also requires a nonempty `checklist` array of questions. The scheduler converts IDs to strings internally. Builders additionally retain source tags, difficulty information where applicable, and raw rewriting output.

The files [alignment.jsonl](../examples/alignment.jsonl) and [quality.jsonl](../examples/quality.jsonl) are hand-written format examples, not samples from paper experiments. A custom prompt stream can use these minimal formats directly.

Check rewritten streams before evaluation:

```bash
python -m prompt_builders.analyze_generated_outputs --input prompt_builders/output/alignment_main/alignment_main.jsonl --output-dir /tmp/alignment-analysis
```

Rewriting can produce malformed or empty outputs. The analysis reports problematic examples; schedulers reject unusable records instead of scoring empty prompts. Regenerate or curate invalid records before a run. Keep existing IDs and text stable once a stream has been used.
