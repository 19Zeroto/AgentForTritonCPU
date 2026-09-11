# Testing Playbook

Before using this playbook, read `agent/rules/testing.md` and `agent/rules/script-writing.md`.

For batch, parallel, resumable, full-suite, or result-summary work, read
`skills/test-suite/SKILL.md` and use its selected entrypoint. Keep the direct
pytest commands below for targeted validation and debugging.

## Common Setup

Run from the repository root after loading the local environment:

```bash
AGENT_DIR="${AGENT_DIR:-$HOME/agent}"
source "${AGENT_DIR}/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"
cd "${AGENT_DIR}/triton-cpu"
```

## Full `test_core.py`

Use pytest-xdist for full `test_core.py` validation:

```bash
OMP_NUM_THREADS=32 python3 -m pytest -q --tb=no -n32 --device=cpu python/test/unit/language/test_core.py -m cpu
```

## Targeted Validation

For several known cases, keep parallel and quiet settings:

```bash
OMP_NUM_THREADS=32 python3 -m pytest -q --tb=no -n32 --device=cpu <case1> <case2> <case3> -m cpu
```

For one or two cases, omit xdist when compilation time is more useful than parallelism:

```bash
OMP_NUM_THREADS=32 python3 -m pytest -q --tb=no --device=cpu <case> -m cpu
```

## Debugging A Single Failure

Use detailed output and single-process execution:

```bash
OMP_NUM_THREADS=32 python3 -m pytest -s -v --device=cpu 'python/test/unit/language/test_core.py::<case>' -m cpu
```

## FlagGems Tests

`FlagGems/tests` do not use Triton `test_core.py` CPU markers.

### Default CPU pipeline

The backend selects the pipeline automatically: payloads containing matmul use
SME; other payloads use SVE. Treat that classification as part of the test
method when choosing cases and interpreting failures.

Recommended multi-worker pairings:

- `-n16`: `OMP_NUM_THREADS=16`
- `-n32`: `OMP_NUM_THREADS=8`

```bash
OMP_NUM_THREADS=16 python3 -m pytest -q --tb=no -n16 FlagGems/tests/<test_file.py>
OMP_NUM_THREADS=8 python3 -m pytest -q --tb=no -n32 FlagGems/tests/<test_file.py>
OMP_NUM_THREADS=32 python3 -m pytest -s -v FlagGems/tests/<test_file.py>::<case>
```

## Benchmark

Run FlagGems benchmarks from the repository root after completing the common setup above. Benchmark files are under `FlagGems/benchmark/test_*.py`; they are separate from Triton `test_core.py` and the tests under `FlagGems/tests`.

### Preflight

- Confirm the target benchmark file and, when needed, its pytest marker.
- Set `OMP_NUM_THREADS` explicitly. Keep it at or below 32.
- Bind the benchmark to a single NUMA node with `numactl` and bind the worker threads to the matching CPU list with `taskset`.
- Use `--level core` for a basic smoke run. Use `--level comprehensive` only when the broader shape set is required.

### CPU and NUMA binding

Replace `<cpu_node>`, `<memory_node>`, and `<cpu_list>` with the target NUMA node and CPU range. The number of CPUs in `<cpu_list>` should match `OMP_NUM_THREADS`:

```bash
OMP_NUM_THREADS=32 numactl --cpunodebind=<cpu_node> --membind=<memory_node> \
  taskset -c <cpu_list> pytest ...
```

For example, the following binds 32 OpenMP threads and memory allocations to NUMA node 3 and CPUs 488--519:

```bash
OMP_NUM_THREADS=32 numactl --cpunodebind=3 --membind=3 \
  taskset -c 488-519 pytest ...
```

### Basic single-operator benchmark

The following command is a low-cost smoke benchmark. Replace `<benchmark_file.py>` and `float32` with the target file and dtype:

```bash
OMP_NUM_THREADS=32 numactl --cpunodebind=<cpu_node> --membind=<memory_node> \
  taskset -c <cpu_list> pytest -q --tb=no \
  FlagGems/benchmark/<benchmark_file.py> \
  --mode operator --level core --dtypes float32 \
  --warmup 5 --iter 5 --metrics latency --record log
```

For multiple dtypes, pass `--dtypes` once per dtype because the option is appendable:

```bash
OMP_NUM_THREADS=32 numactl --cpunodebind=<cpu_node> --membind=<memory_node> \
  taskset -c <cpu_list> pytest -q --tb=no \
  FlagGems/benchmark/<benchmark_file.py> \
  --mode operator --level core \
  --dtypes float16 --dtypes float32 \
  --warmup 5 --iter 5 --metrics latency --record log
```

Use larger `--warmup` and `--iter` values for stable performance measurements after the smoke run.

### Benchmark mode selection

- `--mode operator`: measure the end-to-end operator path. Use this as the default for benchmark runs.
- `kernel` mode: run the legacy cache-clearing `do_bench` path against the full callable; it is not an isolated Triton-kernel measurement and is not recommended in any case.
- `--mode wrapper`: measure the runtime wrapper path.

For kernel performance analysis, keep the benchmark in `operator` mode and use
`perf stat`, `perf record/report`, or another profiling/tracing tool to inspect
generated kernel symbols and hotspots. Do not use `kernel` mode for this purpose.

When comparing benchmark performance, keep the mode, level, dtype, shape file, warmup, and iteration count consistent between runs. The default shape file is `FlagGems/benchmark/core_shapes.yaml`; pass `--shape_file <path>` when a different shape set is required.

### Result interpretation

- A zero pytest exit code means the selected benchmark completed; it does not establish full-suite coverage.
- `SKIPPED` indicates that the selected case was not executed, commonly because of an unsupported dtype, shape, or environment requirement.
- A compile/lowering/runtime failure should be diagnosed separately from a performance result; do not treat it as a valid latency sample.
- `--record log` writes a `result_*.log` file in the current working directory. Keep the command line and log file together when reporting results.
