---
name: flaggems-fusion-metrics
description: Run, resume, merge, or export the FlagGems fusion compute-TFLOPS and memory-bandwidth metric suite. Use for fusion metric coverage, per-operator or full pytest benchmark runs, CSV/run.json reports, and formula Markdown exports. Includes a maintenance reference for infrequent formula or architecture changes.
---

# FlagGems Fusion Metrics

Use `scripts/run_fusion_metrics.py` as the only operational entrypoint. Keep
formula truth in
`$TRITON_REPO_DIR/FlagGems/benchmark/fusion_metrics_formulas.yaml`.

## Run

Activate the configured Python environment before every Python command:

```bash
source ~/myenv/bin/activate
python3 \
  "$AGENT_DIR/agentfortritoncpu/skills/flaggems-fusion-metrics/scripts/run_fusion_metrics.py" \
  --ops silu_and_mul
```

Use `--all` for the full configured set and `--dry-run` to inspect selection,
NUMA bindings, and commands. Results default to
`$AGENT_DIR/logs/fusion_metrics/`; use `--output-dir` to override. Keep
`--omp-threads` at or below 32; the runner sets child `OMP_NUM_THREADS`.

Compute report targets follow the selected pipeline and dtype after applying
the scale `0.2 / 304 * 32`: SME/float32 is 2.52631579 TFLOPS,
SME/bfloat16 is 5.05263158 TFLOPS, and SVE/float32 is 0.71578947 TFLOPS.
Use `--target-tflops` or `FUSION_TARGET_TFLOPS` only to apply an explicit
uniform override; otherwise the runner selects the scaled value per result
row.

Memory report default target bandwidth is `30.72 GB/s`. Override with
`--target-gbps` or `FUSION_TARGET_GBPS` when needed.

When `--dtypes` or `FUSION_DTYPES` explicitly selects a dtype and an operator
run fails, retry that operator with BF16, then retry without `--dtypes` if BF16
also fails. The terminal prints each fallback, and each attempt keeps its own
raw stdout/record log. Runs without an explicit dtype keep the normal default
dtype selection.
Other pipeline/dtype combinations fail preflight until an explicit override is
provided.

Optional defaults come from `scripts/.env`, with precedence
`CLI > scripts/.env > code defaults`. Allowed fields are `FUSION_WARMUP`,
`FUSION_ITER`, `FUSION_MODE`, `FUSION_LEVEL`, `FUSION_DTYPES`, `FUSION_JOBS`,
`FUSION_OMP_THREADS`, `FUSION_NUMA_NODES`, `FUSION_TARGET_TFLOPS`,
`FUSION_TARGET_GBPS`, `FUSION_TIMEOUT`, `FUSION_SHAPE_FILE`, and
`FUSION_OUTPUT_DIR`.

The runner loads the fixed YAML and sets the formula config and selected formula
key explicitly for each pytest child. The formula evaluator binds the actual
benchmark args/kwargs, computes one `logical_flops` or `logical_bytes` result,
and leaves result collection to the existing benchmark path. The runner
preserves raw stdout/record logs and publishes the existing compute, memory,
coverage, and run manifest structures.

## Export formulas

Export documentation without environment, NUMA, or pytest preflight:

```bash
source ~/myenv/bin/activate
python3 \
  "$AGENT_DIR/agentfortritoncpu/skills/flaggems-fusion-metrics/scripts/run_fusion_metrics.py" \
  --export-formulas "$AGENT_DIR/logs/fusion_metrics/formulas.md"
```

Always provide an explicit output file.

For infrequent formula, schema, evaluator, or runner architecture maintenance,
read `references/design.md` completely. Do not load it for routine runs.
