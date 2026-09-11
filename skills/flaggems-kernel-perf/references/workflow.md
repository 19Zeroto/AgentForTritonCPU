# Perf collection and analysis workflow

Use this reference when collecting or regenerating `perf.data`. If valid
per-shape recordings already exist, start at [Analyze a recording](#analyze-a-recording).

## 1. Resolve portable paths

```bash
export AGENT_DIR="${AGENT_DIR:-$HOME/agent}"
ENV_SCRIPT="$AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"
PROFILE_SCRIPT="$AGENT_DIR/AgentForTritonCPU/skills/flaggems-benchmark-profile/scripts/run_profile.py"
ANALYZE_SCRIPT="$AGENT_DIR/AgentForTritonCPU/skills/flaggems-kernel-perf/scripts/analyze_perf.py"
source "$ENV_SCRIPT"
```

Use a task-specific output directory outside both source repositories:

```bash
OUT="${OUT:-$AGENT_DIR/logs/AgentForTritonCPU/flaggems-kernel-perf/<operator>}"
mkdir -p "$OUT"
```

Do not save generated YAML, Triton cache, perf data, or reports in
`AgentForTritonCPU` or `triton-cpu`.

## 2. Confirm environment and permissions

Record the resolved Python, LLVM, repository revision, CPU model, NUMA topology,
thread count, and affinity. Confirm user-space cycle sampling before a long run:

```bash
OMP_NUM_THREADS=32 numactl --physcpubind=<cpu-list> --membind=<node> \
  perf stat -e cycles:u,instructions:u -- /bin/true
```

Stop and report `N/A` if perf returns permission denied, not supported, or not
counted. Do not silently fall back to zero.

## 3. Smoke the benchmark in operator mode

Run from `$TRITON_REPO_DIR`. Keep `OMP_NUM_THREADS` at or below 32 and bind the
same number of CPUs on one NUMA node:

```bash
(
  cd "$TRITON_REPO_DIR" || exit 1
  OMP_NUM_THREADS=32 numactl --physcpubind=<cpu-list> --membind=<node> \
    python3 -m pytest -s FlagGems/benchmark/<test_file.py> \
      --mode operator --level core --warmup 5 --iter 5 \
      --dtypes float32 --metrics latency --record log
)
```

Use `operator` mode even for kernel hotspot work. Perf identifies generated kernel
symbols inside the operator process; benchmark `kernel` mode still measures a full
callable and cache-clearing behavior.

Check the benchmark source, marker, shape YAML, and query output. Do not assume
default shapes survive benchmark-specific filtering. `--case-indices` in the
profiler is zero-based.

## 4. Create an external profiler config

Create the YAML under `$OUT`. Replace the test file and marker; preserve the
chosen dtype, level, and thread policy across repeats:

```yaml
tiers:
  profile: {warmup: 20, iterations: 20}

global:
  mode: operator
  level: core
  default_tier: profile
  metrics: [latency]
  dtypes: [float32]

system:
  triton_cache_dir: null
  clear_triton_cache: true
  numa:
    enabled: false
  cpu_affinity: null
  env_vars:
    OMP_NUM_THREADS: "32"
    MKL_NUM_THREADS: "1"

profiling:
  cpu_memory:
    enabled: false
  cache_metrics:
    enabled: false
  compilation_time:
    enabled: true

tests:
  - test_file: <test_file.py>
    marker: <pytest-marker>
    tier: profile
```

The example uses outer `numactl` binding. Alternatively configure NUMA and CPU
affinity inside the YAML, but do not apply conflicting policies in both places.

Choose warmup and iteration counts by runtime: the process must run long enough
to collect useful kernel samples without turning a large-shape run into an
unbounded job. Record the final values instead of treating `20/20` as universal.

## 5. Dry-run, then collect matched repeats

```bash
OMP_NUM_THREADS=32 numactl --physcpubind=<cpu-list> --membind=<node> \
  python3 "$PROFILE_SCRIPT" \
    --config "$OUT/profile.yaml" \
    --tests <pytest-marker> \
    --output-dir "$OUT" \
    --run-name run-a \
    --run-mode case \
    --case-indices <zero-based-index> \
    --jobs 1 \
    --dtypes float32 \
    --record log \
    --perf-record \
    --perf-record-frequency 99 \
    --perf-record-call-graph dwarf \
    --skip-cpu-mem \
    --dry-run
```

Remove `--dry-run` for collection. Repeat as `run-b` without changing any other
argument. Prefer `dwarf` call chains. If overhead requires `fp`, use `fp` for all
runs and verify that the generated code/runtime preserves usable frame pointers.
Never use `none` for a run that must answer inclusive-tree or remainder questions.

The profiler wraps the complete pytest subprocess. Therefore compile, setup,
input generation, Python, runtime, and kernel execution can all be sampled. This
is intentional for process attribution but is not a kernel-only timing window.
For an exact isolated execution denominator, use a standalone precompiled harness
or explicit profiling markers; label that as a different experiment.

## 6. Identify symbols and call relationships

Locate each case directory and inspect the flat report:

```bash
rg -n "kernel|memrefCopy|malloc|free|sincos|exp|tanh" "$CASE/perf_report.txt"
```

Use the exact generated symbol, which may include rank or specialization suffixes.
Generate a callgraph report if needed:

```bash
perf report --stdio -i "$CASE/perf.data" \
  --sort=dso,symbol > "$OUT/perf_report_callgraph.txt"
```

Only annotate a helper as a kernel child after the call tree shows it beneath the
kernel. A hot flat symbol may belong to input generation or JIT rather than the
kernel tree.

## 7. Analyze a recording

Analyze one run:

```bash
python3 "$ANALYZE_SCRIPT" "$CASE/perf.data" \
  --kernel-symbol <exact-kernel-symbol> \
  --child-symbol <confirmed-child-symbol> \
  --shape '<shape>' \
  --output-dir "$OUT/analysis-run-a"
```

Analyze two matched repeats together:

```bash
python3 "$ANALYZE_SCRIPT" \
  "$CASE_A/perf.data" "$CASE_B/perf.data" \
  --kernel-symbol <exact-kernel-symbol> \
  --child-symbol <confirmed-child-symbol> \
  --shape '<shape>' \
  --output-dir "$OUT/analysis-repeat"
```

The script writes, per run:

- `perf_metadata.txt`: event and `CALLCHAIN` presence.
- `perf_report_flat.txt`: process Self rows with total periods.
- `perf_report_callgraph.txt`: Children/Self and call trees.
- `perf_annotate_<symbol>_period.txt`: period-weighted disassembly.
- `instruction_hotspots_<symbol>.csv` and `mnemonic_summary_<symbol>.csv`.
- `analysis.json`, `commands.txt`, and `report.md`.

With multiple inputs it also writes `repeat_comparison.md`. The comparison checks
event and `CALLCHAIN` presence, but the analyst must still compare commands,
affinity, cache policy, sample quality, rankings, and shares.

## 8. Raw commands used by the analyzer

The essential commands are:

```bash
perf report --stdio -i "$CASE/perf.data" \
  --no-children --show-total-period --sort=dso,symbol

perf report --stdio -i "$CASE/perf.data" --sort=dso,symbol

perf annotate --stdio -i "$CASE/perf.data" \
  --symbol <exact-symbol> \
  --percent-type local-period \
  --show-total-period --full-paths
```

When GNU objdump cannot decode the generated object correctly, pass:

```bash
--objdump "$LLVM_INSTALL_DIR/bin/llvm-objdump"
```
