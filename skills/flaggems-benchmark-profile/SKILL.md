---
name: flaggems-benchmark-profile
description: Profile Triton CPU FlagGems benchmarks with per-operator or per-shape runs, compile-time hooks, CPU and memory sampling, perf stat or perf record, TFLOPS enrichment, and aggregated reports. Use for FlagGems performance investigation, hotspot collection, benchmark evidence, or profiling report generation.
---

# Profile FlagGems Benchmarks

## Workflow

1. Load the `environment` skill.
2. Confirm target benchmark, dtype, mode, level, warmup, iterations, NUMA node,
   CPU affinity, and external output path.
3. Review available options:

   ```bash
   python3 scripts/run_profile.py --help
   ```

4. Start with `--dry-run`; inspect pytest command, binding, config, and external
   output path.
5. Run one operator before `--all`. Use `--run-mode case` only when per-shape
   counters are required.
6. Use `--skip-perf` when perf permissions are unavailable; never present missing
   counters as zero measurements.
7. Report run directory, status, compile time, latency, counters, and invalid or
   skipped samples.

## Resources

- Execute `scripts/run_profile.py`.
- Execute `scripts/tools/perf_heatmap.py` to visualize existing perf output.
- Read `references/usage.md` for full CLI/config/output details.
- Read `references/design.md` only when changing profiler architecture.
- Read `references/tflops.md` when adding or reviewing FLOPs formulas.
- Read `../../agent/references/benchmark-system.md` when interpreting CPU
  pipeline effects.
- Read `references/kunpeng_silu_and_mul_optimization_memo.md` only for that
  historical optimization investigation.
