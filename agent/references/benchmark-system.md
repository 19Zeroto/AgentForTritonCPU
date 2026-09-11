# FlagGems benchmark system reference

FlagGems benchmarks live under `$TRITON_REPO_DIR/FlagGems/benchmark/` and are
driven by pytest. Product code is authoritative when this reference differs
from the current checkout.

## Main components

- `performance_utils.py`: benchmark base classes, measurement, metrics, and
  record output.
- `conftest.py`: pytest options and global benchmark configuration.
- `attri_util.py`: metric/mode enums and fallback shapes.
- `core_shapes.yaml`: shared core shape configuration.
- `test_*.py`: operator-specific inputs, shapes, and metric formulas.

Read `benchmark-shapes.md` before changing shape ownership or fallback order.

## CLI contract

| Option | Behavior |
| --- | --- |
| `--mode kernel|operator|wrapper` | Product default is `kernel`; Triton CPU performance work recommends `operator`. |
| `--level core|comprehensive` | `core` is the smoke set; `comprehensive` adds broader shapes. |
| `--dtypes <dtype>` | Appendable; repeat the option for multiple dtypes. |
| `--metrics <metric>` | Appendable; omit it to use the benchmark class defaults. |
| `--warmup`, `--iter` | Warmup and measured iteration counts. |
| `--shape_file <path>` | Override the default `core_shapes.yaml`. |
| `--record log` | Write machine-readable benchmark records in the working directory. |

The product's `kernel` mode wraps the full callable in its cache-clearing
`do_bench` path. It is not isolated Triton-kernel timing. Use `operator` for
standard comparisons and `perf`/tracing for kernel attribution.

## Metrics

Benchmark-local `get_tflops()` and `get_gbps()` methods are the authoritative
logical-work definitions. Logical FLOPs/bytes are comparison conventions, not
hardware instruction counts or measured DRAM traffic.

Core recorded fields include reference latency, FlagGems latency, speedup,
optional TFLOPS/GB/s, accuracy, and error details. A profiler may provide a
clearly labeled estimate only when the benchmark record lacks a native metric;
it must not overwrite a native value.

## CPU pipeline

The backend selects the pipeline from the payload:

- matmul or batch-matmul payloads use the SME path;
- other payloads use the SVE path.

Treat the actual pipeline as a performance variable verified from IR or compile
logs. Do not invent a pipeline-selection environment variable. Useful evidence
variables include:

| Variable | Purpose |
| --- | --- |
| `TRITON_SHARED_DUMP_PATH=<path>` | Save staged MLIR outside the source tree. |
| `MLIR_ENABLE_DUMP=1` | Print pass-level IR. |
| `TRITON_PRINT_COMPILE_TIME=1` | Record compile-stage timing. |

## Reproducible comparison

Keep mode, level, dtype, shape source, warmup/iterations, toolchain revision,
OpenMP threads, CPU/NUMA binding, and cache policy fixed. CPU and memory nodes
must be selected for the current machine; examples are not portable defaults.

Minimal operator-mode smoke command:

```bash
OMP_NUM_THREADS=<threads> \
numactl --cpunodebind=<cpu-node> --membind=<memory-node> \
taskset -c <cpu-list> \
python3 -m pytest -q --tb=no FlagGems/benchmark/<test_file.py> \
  --mode operator --level core --dtypes float32 \
  --warmup 5 --iter 5 --record log
```

Use `flaggems-benchmark-profile` for data collection and
`flaggems-kernel-perf` for instruction/call-tree attribution. For compute or
memory throughput, inspect the selected benchmark's native `get_tflops()` or
`get_gbps()` implementation and its recorded fields directly; do not maintain
an external formula map or report-conversion skill.
