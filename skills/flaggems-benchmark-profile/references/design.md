# Profiler architecture contract

## Data flow

```text
profiling_config.yaml + CLI overrides
  -> run_profile.py
     -> pytest benchmark subprocess
     -> compile hook / resource monitor / perf stat or record
  -> result_aggregator.py
  -> summary.json + summary.md + raw artifacts
```

`operator` is the maintained comparison mode. The product's `kernel` mode is a
cache-clearing wrapper around the full callable, not isolated kernel timing.

## Component ownership

- `run_profile.py`: orchestration, work-item isolation, commands, signals, and
  artifact manifests.
- `profiling/config.py`: validated YAML model.
- `profiling/compile_hook.py` and `sitecustomize.py`: opt-in compile timing.
- `profiling/monitor.py`: process CPU/RSS sampling.
- `profiling/perf_wrapper.py`: `perf stat`/`perf record` command and parsing.
- `profiling/result_aggregator.py`: record parsing and report generation.
- `profiling/flops.py`: explicitly labeled estimated-FLOPs fallback.
- `profiling/system_info.py`: host and environment snapshot.
- `tools/perf_heatmap.py`: visualization of an existing perf report.

Do not duplicate benchmark execution or metric formulas in the orchestrator.
Benchmark-local metrics win; estimates never replace them.

## Execution modes

- `group`: one benchmark subprocess per configured operator. Use for ordinary
  collection.
- `case`: one subprocess per selected shape. Use when counters or perf data
  must map to an individual shape.

Perf collection is process-wide and can include Python, JIT, runtime, input
generation, and reference work. Kernel-level attribution belongs to the
`flaggems-kernel-perf` skill.

## Configuration contract

The checked-in YAML is a portable, unbound, single-thread safety baseline. A
real performance comparison should use an external copied config with explicit
OpenMP/MKL threads and CPU/NUMA binding chosen for the current host. The run
manifest must preserve the effective configuration.

Parallel jobs and perf collection are mutually exclusive because concurrent
processes make counters and ownership ambiguous. Missing permissions or
unsupported events are `N/A`, never zero.

## Artifact contract

Every run lives outside source repositories and preserves the command,
effective configuration, system information, stdout, record log, compilation
events, optional resource/perf data, and summaries. Interrupted or failed work
must remain distinguishable from a valid zero measurement.
