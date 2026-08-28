# Kunpeng FlagGems silu_and_mul Optimization Memo

## Contents

- [Current Test Scope](#current-test-scope)
- [Key Results](#key-results)
- [Additional Control Result](#additional-control-result)
- [New Optimization Points](#new-optimization-points-identified)
- [Suggested Repair Order](#suggested-repair-order)
- [Next Test Commands](#next-test-commands)
- [Current Working Hypothesis](#current-working-hypothesis)

This memo records the current local profiling observations and a short repair
plan for FlagGems on Kunpeng CPU. The first target is `silu_and_mul`, because it
shows a large gap in the initial benchmark and is representative of fused
pointwise kernels that depend on `tl.exp`.

## Current Test Scope

Main benchmark command used by the profiling tool:

```bash
python3 scripts/run_profile.py \
  --tests silu_and_mul \
  --warmup-override 20 \
  --iter-override 10 \
  --record log \
  --dtypes float32
```

The benchmark itself is `FlagGems/benchmark/test_silu_and_mul.py`.

- Torch baseline: `torch.mul(torch.nn.functional.silu(x), y)`
- Gems implementation: `flag_gems.silu_and_mul`
- Gems forward kernel: `x / (1 + exp(-x)) * y`
- Benchmark class: `GenericBenchmark`
- Current test covers forward only. Backward is implemented in
  `silu_and_mul_grad_kernel`, but it is not covered by this benchmark.

## Key Results

Initial profiler config had:

```yaml
OMP_NUM_THREADS: "1"
MKL_NUM_THREADS: "1"
```

That makes the triton-shared CPU launcher almost single-threaded. The launcher
uses OpenMP over the Triton launch grid, so this setting severely suppresses
Gems throughput.

Initial result with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`:

| Op | Shape | Torch ms | Gems ms | Gems/Torch |
| --- | ---: | ---: | ---: | ---: |
| `silu_and_mul` | `[1073741824]` | 5723.414 | 29066.967 | 0.197x |
| `silu_and_mul` | `[4096,4096]` | 81.051 | 454.040 | 0.179x |
| `silu_and_mul` | `[64,64]` | 0.069 | 0.908 | 0.076x |

After forcing Gems OpenMP parallelism only
(`OMP_NUM_THREADS=32`, `MKL_NUM_THREADS=1`), Gems becomes much faster than the
single-thread Torch baseline on large shapes:

| Op | Shape | Torch ms | Gems ms | Gems/Torch |
| --- | ---: | ---: | ---: | ---: |
| `silu_and_mul` | `[1073741824]` | 5735.873 | 954.773 | 6.008x |
| `silu_and_mul` | `[4096,4096]` | 81.113 | 15.983 | 5.075x |
| `silu_and_mul` | `[64,64]` | 0.064 | 0.728 | 0.088x |

Fairer 32-thread comparison
(`OMP_NUM_THREADS=32`, `MKL_NUM_THREADS=32`, `torch.get_num_threads()==32`):

| Op | Shape | Torch ms | Gems ms | Gems/Torch |
| --- | ---: | ---: | ---: | ---: |
| `silu_and_mul` | `[1073741824]` | 331.660 | 951.852 | 0.348x |
| `silu_and_mul` | `[4096,4096]` | 4.273 | 15.729 | 0.272x |
| `silu_and_mul` | `[64,64]` | 0.066 | 0.922 | 0.072x |

Interpretation:

- The original 5x-15x gap is partly a benchmark configuration problem.
- With fair 32-thread comparison, `silu_and_mul` still trails Torch by roughly
  2.9x-3.7x on large and mid-size shapes.
- Small shape latency remains dominated by launch/runtime overhead.

## Additional Control Result

`add` was tested to distinguish generic pointwise overhead from `tl.exp`
overhead.

Fair 32-thread result:

| Op | Shape | Torch ms | Gems ms | Gems/Torch |
| --- | ---: | ---: | ---: | ---: |
| `add` | `[1073741824]` | 147.595 | 398.063 | 0.371x |
| `add` | `[4096,4096]` | 1.784 | 6.319 | 0.282x |
| `add` | `[64,64]` | 0.022 | 0.972 | 0.023x |

This means the main issue is not only `silu`/`exp`; the generic pointwise CPU
path is already 2.7x-3.6x slower than Torch for a simple binary pointwise op.

## New Optimization Points Identified

### 1. Fix Profiling Thread Configuration First

Current profile results can be misleading when `OMP_NUM_THREADS=1`. For
throughput tests on the current NUMA binding (`taskset -c 456-487`), use:

```bash
OMP_NUM_THREADS=32
MKL_NUM_THREADS=32
```

The profiler should record at least:

- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`
- `torch.get_num_threads()`
- CPU affinity
- NUMA node

The result table should label whether the comparison is single-thread,
Gems-only multithread, or fair multithread.

### 2. Add a Kunpeng CodeGenConfig

`FlagGems/src/flag_gems/utils/codegen_config_utils.py` has no
`vendors.KUNPENG` entry. Kunpeng falls back to the NVIDIA config:

```python
CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=int(triton.__version__[0]) < 3,
)
```

For CPU pointwise this `max_tile_size=512` is too small. It creates too many
small units and amplifies launch, loop, temporary-buffer, and copy overhead.

First sweep candidates:

- `max_tile_size=4096`
- `max_tile_size=8192`
- `max_tile_size=16384`
- compare `prefer_block_pointer=True/False`
- compare `prefer_1d_tile=True/False`

### 3. Eliminate Temporary Buffer + memrefCopy in Pointwise Lowering

Generated LLVM IR for `add` and `silu_and_mul` shows the pattern:

- allocate a 512-float temporary buffer
- scalar loop over 512 elements
- write results into the temporary buffer
- `memrefCopy` temporary buffer to output
- free the temporary buffer

This is a direct performance problem for all pointwise kernels. The generated
code should store directly to the real output tensor whenever the output is
contiguous or when the pointer analysis can prove the direct write is legal.

Compiler remarks also show repeated `PtrAnalysis` failures:

```text
PtrAnalysis: pointer is not replace with tts.make_tptr so loadOp cannot be rewritten
PtrAnalysis: Failed to rewrite LoadOp
PtrAnalysis: scalar storeOp will not be rewritten
PtrAnalysis: Failed to rewrite StoreOp
```

This is a strong hint that direct tptr-based load/store conversion is not
triggering for the current pointwise path.

### 4. Add a Contiguous Fast Path for silu_and_mul

Most benchmark shapes are same-shape contiguous tensors. For that common case,
`silu_and_mul` can bypass the fully dynamic pointwise wrapper:

- flatten contiguous input/output tensors
- use one-dimensional indexing
- direct load from `x` and `y`
- direct store to `out`
- avoid stride-order computation
- avoid block pointer path
- use a larger tile

This can be implemented either as:

- a Kunpeng-specific fused op under `runtime/backend/_kunpeng/fused`, or
- a generic contiguous fast path in `pointwise_dynamic`.

The dedicated `silu_and_mul` path is lower risk for initial validation because
it has a narrow behavioral surface.

### 5. Improve Vectorization and exp Lowering

The `silu_and_mul` generated LLVM IR currently contains a scalar loop and scalar
FMA/floor polynomial sequence for `exp`. There is no clear SVE vector loop in
the inspected final LLVM IR.

After direct output storage is fixed, inspect whether the final code lowers to
SVE vector operations. If not, add or adjust lowering so pointwise maps to
vector transfer/arith loops on AArch64/SVE.

For `silu_and_mul`, the math path is:

```python
x_fp32 = x.to(tl.float32)
x_silu = x_fp32 / (1.0 + tl.exp(-x_fp32))
out = x_silu * y
```

Potential math-specific improvements:

- use vectorized SLEEF/ArmPL/SVE exp where available
- verify current polynomial is scalar because of failed vector lowering, not
  because `tl.exp` itself lacks a vector path
- consider a Kunpeng-specific sigmoid/silu approximation only after correctness
  tolerance is agreed

### 6. Add Small-Shape Fallback

For `[64,64]`, Gems remains around `0.7-1.0 ms`, while Torch is around
`0.02-0.07 ms`. This is not mainly arithmetic; it is launch/runtime overhead.

Add a threshold fallback for small tensors:

```text
if numel < threshold:
    use torch baseline or a lightweight CPU fallback
```

Candidate thresholds to sweep:

- `16K`
- `64K`
- `256K`
- `1M`


## Suggested Repair Order

1. Update profiler config/reporting so thread settings are explicit and fair.
2. Add a Kunpeng `CodeGenConfig` and sweep tile size/block-pointer/1D options.
3. Fix pointwise lowering so contiguous outputs write directly, without
   temporary buffer + `memrefCopy`.
4. Add a dedicated contiguous fast path for `silu_and_mul`.
5. Re-check generated LLVM IR for SVE/vector loops.
6. Optimize `tl.exp`/sigmoid/silu math path if vectorization is still not enough.
7. Add small-shape fallback.
8. Add backward benchmark if training performance matters.

## Next Test Commands

Fair 32-thread `silu_and_mul`:

```bash
cd $AGENT_DIR/AgentForTritonCPU/skills/flaggems-benchmark-profile
source "$AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"
cd $TRITON_REPO_DIR/FlagGems/benchmark

OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 PYTHONDONTWRITEBYTECODE=1 \
numactl --cpunodebind=3 --membind=3 taskset -c 456-487 \
python3 -m pytest test_silu_and_mul.py -s -m silu_and_mul \
  --record log \
  --mode kernel \
  --level core \
  --warmup 20 \
  --iter 10 \
  --dtypes float32
```

Per-shape hotspot top10 with Linux perf:

```bash
cd $AGENT_DIR/AgentForTritonCPU/skills/flaggems-benchmark-profile
source "$AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"

python3 scripts/run_profile.py \
  --tests silu_and_mul \
  --warmup-override 20 \
  --iter-override 10 \
  --record log \
  --dtypes float32 \
  --run-mode case \
  --case-indices 4 \
  --perf-record
```

This writes the following files under the selected case directory:

- `perf.data`
- `perf_report.txt`
- `perf_top10.txt`
- `perf_heatmap.html` / `perf_heatmap.svg` can be generated later with:

  ```bash
  python3 scripts/tools/perf_heatmap.py \
    "$AGENT_DIR/logs/AgentForTritonCPU/flaggems-benchmark-profile/<run_name>/silu_and_mul/case_004"
  ```

Use `case_001` (`[64,64]`) to inspect small-shape runtime overhead, and
`case_004` (`[1024,1024,1024]`) or `case_000` (`[1073741824]`) to inspect the
large-shape compute kernel. The small smoke run on `case_001` showed Python
interpreter and GC symbols at the top, which is consistent with launch/runtime
overhead dominating small tensors.

Useful sweeps after the first patch:

- `OMP_NUM_THREADS=1,2,4,8,16,32`
- `max_tile_size=512,4096,8192,16384`
- `prefer_1d_tile=True/False`
- `prefer_block_pointer=True/False`
- compare `silu`, `mul`, `add`, and `silu_and_mul`

## Current Working Hypothesis

For `silu_and_mul`, the biggest actionable direction is not changing the Python
formula first. The current bottleneck is likely:

1. pointwise CPU codegen using a GPU-like default config on Kunpeng,
2. scalar 512-element tile loops,
3. temporary output buffer plus `memrefCopy`,
4. missing or ineffective SVE vector lowering,
5. fixed launch overhead on small tensors.

Once direct contiguous vector output is fixed, `tl.exp`/silu math can be
profiled as the next isolated bottleneck.
