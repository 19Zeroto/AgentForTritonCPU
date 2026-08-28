# FlagGems fusion metrics design

## Objective

Keep every operator's logical FLOPs or logical bytes formula beside its input
construction in the corresponding FlagGems benchmark class. Record the
resulting `tflops`, `gbps`, and `gbps_base` values in the normal benchmark log.
Keep this skill responsible only for serial execution, log extraction, and CSV
presentation.

Logical metrics define a stable comparison convention; they are not hardware
instruction or DRAM traffic counters.

## Data flow

```text
benchmark-local get_tflops/get_gbps
        -> FlagGems record log
        -> fusion_metrics.py
        -> compute CSV / memory CSV / coverage / run manifest
```

There is no formula YAML, fixed metric map, benchmark-process injection, or
external compute/memory classification map.

## Benchmark ownership

Each supported benchmark class:

- copies its default metric list and appends exactly one of `tflops` or `gbps`;
- implements the matching method locally;
- returns logical FLOPs from `get_tflops()`;
- returns `logical_bytes / latency_ms / 1e6` from `get_gbps()`;
- performs any value-dependent inspection outside the measured latency call.

Passing only `latency_base`, `latency`, and `speedup` through `--metrics`
disables the optional throughput method for that run.

### Compute benchmarks

- `flash_attention_forward`
- `flash_attn_varlen_func`
- `flash_mla`
- `flash_mla_sparse_fwd`
- `sparse_mla_fwd_interface`

### Memory benchmarks

- `apply_repetition_penalties`
- `apply_rotary_pos_emb`
- `concat_and_cache_mla`
- `cross_entropy_loss`
- `dgeglu`
- `dreglu`
- `fused_add_rms_norm`
- `geglu`
- `gelu_and_mul`
- `get_scheduler_metadata`
- `instance_norm`
- `moe_sum`
- `reglu`
- `reshape_and_cache`
- `reshape_and_cache_flash`
- `rwkv_ka_fusion`
- `rwkv_mm_sparsity`
- `silu_and_mul`
- `silu_and_mul_out`
- `skip_layer_norm`

`concat_and_cache_mla` and `cross_entropy_loss` retain their current invocation
kwargs in their local benchmark instances because the shared `get_gbps()`
interface receives positional args and latency only.

`rwkv_mm_sparsity` uses the actual input non-zero count. For `k[M]`, `v[M,N]`,
and output `o[N]`:

```text
logical_bytes = B(k) + count_nonzero(k) * N * element_size(v) + B(o)
```

Cache that logical-byte result for the two `gbps_base`/`gbps` calls associated
with one input. Do not use a fixed 5% density estimate.

The disabled operators remain outside the runner suites:

- `moe_align_block_size_triton`: benchmark output capacity blocker.
- `topk_softmax`: benchmark arguments do not match the product interface.
- `weight_norm`: accuracy coverage remains skipped.

## Runner

Run at most one benchmark subprocess at a time. Validate NUMA selections,
limit `OMP_NUM_THREADS` to 32, and do not force the compiler pipeline. Use the
benchmark's default metric list rather than passing `--metrics`.

Execute each operator in a short system temporary directory so record filenames
remain bounded. Link an explicitly supplied shape file into that directory,
archive record/stdout logs under the run directory, and remove the temporary
directory immediately afterward. Do not publish a persistent `work/` tree.

The run manifest records execution facts: mode, level, warmup, iteration count,
dtype, NUMA/CPU binding, command, exit status, and log paths.

## Report extraction

Classify each successful record row only from the present throughput field:

- numeric `tflops` -> compute row;
- numeric `gbps` -> memory row;
- neither or both -> coverage error, no data row.

Use recorded latency and throughput to reconstruct presentation-only logical
amounts while preserving existing CSV columns:

```text
logical_flops = tflops * latency_ms * 1e9
logical_bytes = gbps * latency_ms * 1e6
```

Read `gbps_base` directly when present; derive it from logical bytes and Torch
latency only as compatibility fallback. Target TFLOPS/GB/s affect presentation
columns only and never require benchmark reruns.

Write reports to a timestamped snapshot first, then atomically publish the
three top-level CSVs and `run.json`. A failed extraction must not partially
replace an existing top-level report.

## Output

```text
<run-dir>/
├── records/
├── stdout/
├── run.json
└── report/
    ├── fused_compute_tflops.csv
    ├── fused_non_compute_bandwidth.csv
    ├── coverage.csv
    ├── run.json
    └── runs/<timestamp>/
```
