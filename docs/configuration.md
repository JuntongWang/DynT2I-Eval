# Configuration and recovery

Copy `configs/runtime.example.json` to `configs/runtime.json`. Unused model and evaluator entries can remain placeholders. Configure every checkpoint and external implementation used by your selected dimension and models.

| Key | Purpose |
| --- | --- |
| `prompt_builder_model_path` | Local Qwen3.5 checkpoint for prompt rewriting |
| `model_paths` | Local checkpoint directories keyed by registry name |
| `model_envs` | Existing Conda environment names for generation workers |
| `external_code` | Local source trees for specialized adapters and judges |
| `devices` | Device placement within each evaluator process |
| `api_urls` | Evaluator HTTP endpoints |

Paths support `~`; relative checkpoint and source paths are resolved against the repository root. Configuration file paths are resolved against the launching directory.

Configuration precedence is an explicit scheduler `--config`, then `DYNT2I_EVAL_CONFIG`, then the legacy `DYNAMIC_AIGI_BENCH_CONFIG`, then `configs/runtime.json`. The example file supplies unconfigured defaults when no local file exists. An explicitly selected missing file is an error.

Use `DYNT2I_EVAL_CONFIG` when builders, services, and schedulers should share a configuration. A scheduler's `--config` is forwarded to its generation workers; services started separately still need the environment variable. An explicit configuration also controls the scheduler's default API URL. `--api-url` overrides that URL.

## Scheduler options

All dimensions support `--models`, `--gpu-a`, `--gpu-b`, `--prompts`, `--output-dir`, `--state-file`, `--micro-batch-size`, `--rounds`, `--worker-timeout`, `--eval-retries`, `--api-url`, and `--config`. Use `--help` for values. Perceptual quality adds `--iqa-samples` (default 4); aesthetics adds `--tie-threshold` (default 3.5).

Defaults use a micro-batch of 15, a one-hour round timeout, and three evaluation attempts per prompt. Individual HTTP requests have dimension-specific timeouts. Increase the round timeout for slow checkpoints. GPU guard allocation is an optional advanced setting, disabled by default; choose its size for your hardware if enabling it.

## Resume and model injection

Reuse the same prompt file, state file, output directory, checkpoints, and model names to resume. Prompt IDs are unique within a run and must retain their meaning. Append fresh prompt IDs to extend the stream; the scheduler reloads the prompt file when it needs more prompts.

Add a registered name to `--models` to introduce it into a saved state. New model records receive the initial prior. Only selected names participate in scheduling; saved records for omitted models remain available in history and the leaderboard. A model that is not yet registered needs an adapter as described in [models.md](models.md).

For an independent evaluation, choose both a new state file and a new output directory. Replacing the prompt file while keeping cached images or used IDs can mix incompatible runs. Keep immutable copies of checkpoints, input streams, runtime settings, and package versions for reproducibility.

## Failures

Invalid prompts, duplicate IDs, failed workers, missing or corrupt images, and exhausted evaluator retries raise an error. A failed round does not consume its prompts or update ratings. Stop, resolve the cause, and restart with the same paths. Valid cached images are reused; remove a failed image file before retrying if it is corrupt. State writes use a temporary file and atomic replacement. As with ordinary file-based research pipelines, avoid running two schedulers against the same output directory or state file.

Automatic model freezing is disabled, matching the original continual evaluation configuration. The uncertainty floor is not a stopping criterion. Use `--rounds` or prompt exhaustion to bound a run.
