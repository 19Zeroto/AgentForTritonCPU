# Profiler usage

## Prepare

Load the shared environment and inspect the current CLI:

```bash
AGENT_DIR="${AGENT_DIR:-$HOME/agent}"
source "$AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"
python3 "$AGENT_DIR/AgentForTritonCPU/skills/flaggems-benchmark-profile/scripts/run_profile.py" --help
```

Copy `scripts/profiling_config.yaml` to `$AGENT_DIR/cache` or `/tmp` before
changing host-specific binding. Do not commit CPU IDs or NUMA nodes from one
machine as shared defaults.

Example external configuration fragment:

```yaml
system:
  numa:
    enabled: true
    cpunodebind: <cpu-node>
    membind: <memory-node>
  cpu_affinity: "<cpu-list>"
  env_vars:
    OMP_NUM_THREADS: "<threads>"
    MKL_NUM_THREADS: "<threads>"
```

## Collect

Start with a dry run and one operator:

```bash
python3 scripts/run_profile.py \
  --config <external-config.yaml> \
  --tests <operator> \
  --output-dir "$AGENT_DIR/logs/AgentForTritonCPU/flaggems-benchmark-profile" \
  --dry-run
```

Use `--all` only after the single-operator command is correct. Use
`--run-mode case --case-indices <indices>` for per-shape perf evidence. Keep
`--jobs 1` whenever perf is enabled; use `--skip-perf` when permissions are not
available.

The most important overrides are:

- `--warmup-override` and `--iter-override` for smoke versus stable runs;
- `--dtypes`, repeated or passed as accepted by the current CLI;
- `--record log` to retain product benchmark records;
- `--perf-record` for later symbol/call-tree analysis;
- `--skip-cpu-mem`, `--skip-perf`, or `--skip-compile-hook` when a collector is
  intentionally disabled.

## Interpret

- Compare only runs with matching toolchain, shapes, mode, iterations, binding,
  and cache policy.
- Treat the benchmark record as the latency and native-metric source.
- Inspect `tflops_source` and `flops_formula_id` before using an estimated
  throughput value.
- `perf.data` represents the whole subprocess. Use the kernel-perf skill before
  attributing cycles to a generated kernel or child symbol.
- Keep raw stdout and invalid/skipped rows; do not silently reduce them to a
  successful mean.

Run output contains `summary.json`, `summary.md`, `system_info.txt`, the
effective configuration, and per-operator or per-case directories. Optional
files appear only when their collectors ran successfully.
