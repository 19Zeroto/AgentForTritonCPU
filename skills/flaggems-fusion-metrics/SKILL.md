---
name: flaggems-fusion-metrics
description: Run FlagGems fusion benchmarks and export compute or memory CSV reports directly from benchmark record logs containing tflops or gbps. Use for serial fusion benchmark runs, throughput report generation, or selected-operator report refreshes.
---

# FlagGems fusion metrics

Treat benchmark-local `get_tflops()` and `get_gbps()` implementations as the
metric source of truth. The selected metric is enabled by default in each
supported benchmark. Use `fusion_metrics.py` only to extract recorded metrics
and arrange CSV reports; do not calculate operator formulas in the skill.

Read `references/design.md` before changing benchmark metric formulas, report
columns, operator classification, or runner behavior.

Activate the configured environment before Python commands:

```bash
source ~/myenv/bin/activate
```

Run the configured operators serially:

```bash
python3 "$AGENT_DIR/AgentForTritonCPU/skills/flaggems-fusion-metrics/scripts/run_fusion_benchmarks.py" \
  --suite all --mode operator --level core --warmup 5 --iter 5 \
  --cpu-node 1 --mem-node 1 --omp-threads 32
```

Use `--suite compute|memory|all`, or replace `--suite` with `--ops <names>`.
Use `--dry-run` to inspect commands. The runner uses one temporary working
directory per operator, archives logs below
`$AGENT_DIR/logs/fusion_metrics/benchmark_runs/`, removes the temporary
directory, and then generates `<run-dir>/report`.

Export a report from existing record logs:

```bash
python3 "$AGENT_DIR/AgentForTritonCPU/skills/flaggems-fusion-metrics/scripts/fusion_metrics.py" \
  report --logs <record.log-or-run-dir>
```

Refresh selected operators in an existing report:

```bash
python3 "$AGENT_DIR/AgentForTritonCPU/skills/flaggems-fusion-metrics/scripts/fusion_metrics.py" \
  update-report --logs <new-run-dir> --ops silu_and_mul \
  --output-dir <existing-report-dir>
```

Classify rows from the recorded fields: `tflops` means compute and `gbps`
means memory. Preserve the existing CSV filenames and column order:

- `fused_compute_tflops.csv`
- `fused_non_compute_bandwidth.csv`
- `coverage.csv`
- `run.json`

To run a benchmark without its default throughput metric, explicitly request
only `latency_base`, `latency`, and `speedup` with repeated `--metrics` options.
Do not add formula maps, formula evaluation, metric injection, pipeline forcing,
automatic dtype fallback, retries, or concurrent benchmark jobs.
