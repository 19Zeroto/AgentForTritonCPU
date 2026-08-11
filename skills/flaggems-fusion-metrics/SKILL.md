---
name: flaggems-fusion-metrics
description: Export FlagGems fusion compute and memory metrics from benchmark record logs without changing the measured benchmark path.
---

# FlagGems fusion metrics

This skill is a post-processing workflow. Benchmark processes record only
`latency_base`, `latency`, `speedup`, dtype, and complete `shape_detail`.
Logical FLOPs/bytes are calculated outside the product process and materialized
in `scripts/fusion_metrics_map.json`.

Read `references/design.md` for architecture maintenance or formula changes.

Activate the configured environment before Python commands:

```bash
source ~/myenv/bin/activate
```

Run a single-job serial benchmark (multiple `--ops` are executed one at a time):

Omitting `--ops` runs the full operator list maintained in
`scripts/run_fusion_benchmarks.py`. Use `--suite compute` or
`--suite memory` to run one category; `--suite all` is the default.

```bash
python3 "$AGENT_DIR/AgentForTritonCPU/skills/flaggems-fusion-metrics/scripts/run_fusion_benchmarks.py" \
  --suite all --mode operator --level core --warmup 5 --iter 5 \
  --cpu-node 1 --mem-node 1 --omp-threads 32
```

The runner composes pytest commands, applies explicit CPU/memory NUMA binding,
and stores stdout/record logs plus `run.json` under
`$AGENT_DIR/logs/fusion_metrics/benchmark_runs/`. After the serial benchmark
batch finishes, it invokes `fusion_metrics.py report` and writes the report to
`<run-dir>/report`. It does not evaluate formulas in the benchmark process,
add `tflops`/`gbps`, set pipeline/cache variables, retry, or change dtype. Use
`--dry-run` to inspect commands without executing or reporting.

CPU bindings use numactl physical CPU ranges, for example
`--cpu-list 300-331`; memory bindings accept numactl node lists such as
`--mem-node 20,22,23,27`. The selected nodes must exist and expose memory on
the current host.

Build or incrementally update the fixed map from existing logs:

```bash
python3 "$AGENT_DIR/AgentForTritonCPU/skills/flaggems-fusion-metrics/scripts/fusion_metrics.py" \
  build-map --logs <record.log-or-run-dir> --output-map <map.json>
python3 "$AGENT_DIR/AgentForTritonCPU/skills/flaggems-fusion-metrics/scripts/fusion_metrics.py" \
  build-map --logs <new.log-or-run-dir> --ops silu_and_mul \
  --update-map <map.json> --output-map <map.json>
```

Export a report, or replace only selected operators in an existing report:

```bash
python3 "$AGENT_DIR/AgentForTritonCPU/skills/flaggems-fusion-metrics/scripts/fusion_metrics.py" \
  report --logs <run-dir>
python3 "$AGENT_DIR/AgentForTritonCPU/skills/flaggems-fusion-metrics/scripts/fusion_metrics.py" \
  update-report --logs <new-run-dir> --ops silu_and_mul \
  --output-dir <existing-report-dir>
```

For `report`, the map defaults to `scripts/fusion_metrics_map.json` and the
output defaults to `<run-dir>/report`.  The report assigns `SME` to compute
operators and `SVE` to memory operators.  Use `--map`, `--output-dir`, or
`--pipeline SME|SVE` only when overriding these defaults.

Reports contain the existing compute/memory CSV columns, `coverage.csv`, and
`run.json`. The report stage only queries the map and derives presentation
columns; it does not execute formulas, torch, FlagGems, or pytest.
