# silu_and_mul Performance Optimization Points

## Contents

- [Baseline Environment](#baseline-environment)
- [Current Findings](#current-findings)
- [Perf Findings](#perf-findings)
- [Torch Comparison](#torch-comparison)
- [Compile-Time Impact](#compile-time-impact)
- [Optimization Directions](#optimization-directions)
- [Dense 1D Fast Path Experiment](#dense-1d-fast-path-experiment)
- [Perf Profiling Plan](#perf-profiling-plan)
- [Follow-Up Experiments](#2026-06-30-follow-up-experiments)
- [Direction Status](#direction-status-after-100100-retests)

Date: 2026-06-27
Target: `FlagGems/benchmark/test_silu_and_mul.py` on Kunpeng CPU backend

## Baseline Environment

- Environment setup: `source $AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh`
- Fixed benchmark thread count: `OMP_NUM_THREADS=32`
- Stable single-dtype baseline: `--warmup 100 --iter 100`
- Latest full/comprehensive sweep: `--warmup 10 --iter 10`
- NUMA/core binding used for main runs: node3, cores `456-487`
- Large-shape focus:
  - `[1073741824]`
  - `[1024, 1024, 1024]`
  - `[10000, 65536]`
  - `[100, 65536, 100]`

## Current Findings

1. Kunpeng currently falls back to NVIDIA-style `CodeGenConfig`.
   - Effective config:
     `CodeGenConfig(max_tile_size=512, max_grid_size=(65536, 65536, 65536), max_num_warps_per_cta=32, prefer_block_pointer=True, prefer_1d_tile=False)`
   - There is no dedicated `vendors.KUNPENG` entry in `FlagGems/src/flag_gems/utils/codegen_config_utils.py`.

2. Large contiguous same-shape inputs are collapsed to rank-1 by `pointwise_dynamic`.
   - Original 2D/3D shape rank is not the primary generated-kernel distinction for contiguous inputs.
   - The dominant dynamic difference is `numel`, tile size, `tiles_per_cta`, dtype lowering, and runtime memory state.

3. Default large-shape behavior is dtype-sensitive.
   - f16 is often near Torch, but noisy.
   - f32 commonly lags Torch.
   - bf16 is consistently the weakest dtype and is not fixed by simple tile-size tuning.

4. `max_tile_size` tuning is not monotonic.
   - For `[10000, 65536]`, `tile_size=128` sometimes improves individual runs, especially through `prefer_1d_tile=True`, but paired rechecks show it is not stable enough for a global default.
   - `tile_size=1024 + prefer_block_pointer=True` repeatedly looked promising for f32 and sometimes bf16.
   - `tile_size=2048` tends to regress and changes generated metadata to `num_warps=8`.
   - `prefer_block_pointer=False` without `prefer_1d_tile=True` was generally poor, especially for bf16.

5. Fixed launch/wrapper overhead dominates tiny shapes.
   - Small shapes such as `[64, 64]` are not representative for large model-like pointwise throughput.
   - Large shapes enter grid-stride-loop codegen, e.g. `[10000, 65536]` has:
     - tile 128: `tiles_per_cta=79`
     - tile 512: `tiles_per_cta=20`
     - tile 1024: `tiles_per_cta=10`
     - tile 2048: `tiles_per_cta=5`

## Perf Findings

Current perf focus:

- Shape: `[10000, 65536]`
- Dtypes: `float32`, `bfloat16`
- Runner: Gems-only `flag_gems.silu_and_mul`, no Torch reference in the hot loop
- Binding: node3, cores `456-487`
- Cache roots:
  - `TRITON_CACHE_DIR=/tmp/triton-cache-silu-perf-current`
  - `FLAGGEMS_CACHE_DIR=/tmp/flaggems-cache-silu-perf-current`

1. The dominant hotspot is the generated JIT kernel.
   - Flat `perf record` samples:
     - f32: `silu_and_mul_kernel_kernel_rank_1` is about 65% self samples.
     - bf16: `silu_and_mul_kernel_kernel_rank_1` is about 77% self samples.
   - Call-graph runs also put about 88-89% child time under the OpenMP microtask path that enters this JIT kernel.
   - Python, benchmark harness, Torch reference, and wrapper code are not the large-shape primary bottleneck.

2. OpenMP/runtime overhead is visible but secondary for this large shape.
   - `libomp.so`, scheduling, and `sched_yield` symbols together account for roughly low-double-digit sample share.
   - This matters more for tiny shapes and repeated launch overhead, but the large-shape throughput bottleneck is inside the generated kernel body.

3. The run is not cache-miss dominated.
   - `perf stat` f32: IPC about 1.38, cache miss rate about 0.575%.
   - `perf stat` bf16: IPC about 1.25, cache miss rate about 0.128%.
   - bf16 executes many more instructions/task-clock than f32 despite lower byte traffic, so the gap is more likely lowering/codegen pressure than raw cache misses.

4. f32 hot instructions point to vector math plus spill/store pressure.
   - Annotate shows a large `<512 x float>` exp/div lowering sequence, including `llvm.fma.v512f32`, `llvm.floor.v512f32`, and `fdiv <512 x float>`.
   - The hottest annotated instruction is a scalar load/store tail/copy pattern (`ldr s0` / `str s0`), while many SVE `str z*` stack stores each take smaller but repeated sample share.
   - This supports reducing excessive vector width/register pressure or improving the dense contiguous path.

5. bf16 adds a separate lowering problem on top of the f32 math body.
   - TTSharedIR explicitly loads bf16, extends to f32, computes in f32, then truncates back to bf16.
   - The transform sequence includes `arith-emulate-unsupported-floats` with `source-types=...bf16 target-type=f32`.
   - LLVM IR contains repeated `<32 x bfloat>` load/insertelement/shuffle patterns and final bf16 stores.
   - Perf annotate shows many q/h load/store and stack spill/reload samples.
   - This is a concrete repair point, not just a benchmark artifact.

6. Generated artifacts used for this profile:
   - f32:
     `/tmp/triton-cache-silu-perf-current/8bRkc7F1DMBT6Im6pcLVlxU--2JgwzbihZ_Dpv3KUCA/silu_and_mul_kernel_kernel_rank_1.{ttsharedir,llir,obj}`
   - bf16:
     `/tmp/triton-cache-silu-perf-current/eH2viBBLYo9Ln_AEdRc4ukS2n69SzxRyO4-b_DB-658/silu_and_mul_kernel_kernel_rank_1.{ttsharedir,llir,obj}`

## Torch Comparison

The important comparison is not only between Triton CPU codegen variants, but against the Torch baseline from the benchmark:

- Benchmark form:
  `torch.mul(torch.nn.functional.silu(x), y)`
- This is not fused at Python level, but PyTorch dispatches mature CPU vectorized kernels for both `silu` and `mul`.
- A standalone same-shape comparison on `[10000, 65536]` with `randn` inputs showed:
  - f32: Torch about 314 ms, Gems about 531 ms.
  - bf16: Torch about 231 ms, Gems about 778 ms.
  - The exact numbers are noisy, but they match the benchmark direction: Gems loses badly, especially for bf16.

Torch-only perf on the same shape with `ones` inputs shows:

1. f32 Torch hotspots:
   - about 35% in PyTorch `VectorizedLoop2d` silu kernel,
   - about 27% in PyTorch `VectorizedLoop2d` mul kernel,
   - about 23% in `Sleef_finz_expf4_u10advsimd`.

2. bf16 Torch hotspots:
   - about 40% in PyTorch `VectorizedLoop2d` bf16 silu kernel,
   - about 20% in PyTorch `VectorizedLoop2d` bf16 mul kernel,
   - about 34% in `Sleef_finz_expf4_u10advsimd`.

Interpretation:

- Torch wins because its unfused kernels are tight CPU vector loops with well-tested BFloat16 and SLEEF exp paths.
- Gems fuses the expression, but current generic Triton CPU lowering generates a large 512-element vector body, extra materialization/tail paths, stack spills, and for bf16 an emulated `bf16 -> f32 -> bf16` path.
- Fusion is only useful if the generated loop is as efficient as, or close to, Torch's per-op vector kernels. Here, the fused kernel removes one memory pass but pays more in instruction count, register pressure, conversion/shuffle overhead, and OpenMP/runtime overhead.
- For bf16 specifically, Torch appears to keep the BFloat16 vector path much cleaner, while Gems expands and shuffles bf16 through f32 computation. This explains why bf16 loses far more than f32.

### Official Float32 Comprehensive Shape Ranking

Command:

`source $AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh && export OMP_NUM_THREADS=32 && export TRITON_CACHE_DIR=/tmp/triton-cache-silu-official-comprehensive-f32-20260629a && numactl --cpunodebind=3 --membind=3 taskset -c 456-487 pytest -s FlagGems/benchmark/test_silu_and_mul.py --mode operator --level comprehensive --dtypes float32 --warmup 3 --iter 3 --record log`

This run uses benchmark-provided core/comprehensive shapes, not a custom shape file or standalone runner.

Float32 `silu_and_mul` sorted by speedup:

| Shape | Numel | Torch ms | Gems ms | Speedup | Gems/Torch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[64, 64]` | 4,096 | 0.043 | 0.567 | 0.077 | 13.06x |
| `[10000, 1]` | 10,000 | 0.071 | 0.621 | 0.114 | 8.76x |
| `[100, 1, 100]` | 10,000 | 0.072 | 0.580 | 0.125 | 8.03x |
| `[1024, 1024, 1024]` | 1,073,741,824 | 332.921 | 488.479 | 0.682 | 1.47x |
| `[1073741824]` | 1,073,741,824 | 345.860 | 489.302 | 0.707 | 1.41x |
| `[64, 512, 512]` | 16,777,216 | 9.559 | 12.526 | 0.763 | 1.31x |
| `[268435456]` | 268,435,456 | 83.850 | 105.417 | 0.795 | 1.26x |
| `[10000, 65536]` | 655,360,000 | 252.266 | 297.360 | 0.848 | 1.18x |
| `[100, 65536, 100]` | 655,360,000 | 254.618 | 298.455 | 0.853 | 1.17x |
| `[4096, 4096]` | 16,777,216 | 9.935 | 7.165 | 1.387 | 0.72x |
| `[100, 256, 100]` | 2,560,000 | 8.221 | 1.928 | 4.263 | 0.23x |
| `[10000, 256]` | 2,560,000 | 8.530 | 2.000 | 4.265 | 0.23x |

Interpretation:

- Tiny shapes are dominated by fixed wrapper/dispatch/runtime overhead and should not drive the main large-shape codegen fix.
- The largest absolute and stable large-shape gaps are `[1024, 1024, 1024]` and `[1073741824]`, both at about 1G elements and about 143-156 ms slower than Torch in this run.
- `[268435456]` also loses, but less severely. The two 655M official comprehensive shapes lose only about 17-18% in this run, although earlier longer custom-shape runs showed more noise.
- `[64, 512, 512]` is inconsistent across official runs: this comprehensive 3/3 run loses, while earlier core runs sometimes had Gems winning. Treat it as a variance/recheck target rather than the first optimization anchor.

### Official Float32 Component Benchmarks

Command:

`source $AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh && export OMP_NUM_THREADS=32 && export TRITON_CACHE_DIR=/tmp/triton-cache-silu-components-official-f32-20260629a && numactl --cpunodebind=3 --membind=3 taskset -c 456-487 pytest -s FlagGems/benchmark/test_silu.py::test_silu FlagGems/benchmark/test_mul.py::test_mul --mode operator --level comprehensive --dtypes float32 --warmup 3 --iter 3 --record log`

Core 1G-shape component results:

| Op | Shape | Torch ms | Gems ms | Speedup | Gems/Torch |
| --- | --- | ---: | ---: | ---: | ---: |
| `silu` | `[1073741824]` | 198.850 | 228.305 | 0.871 | 1.15x |
| `silu` | `[1024, 1024, 1024]` | 193.488 | 227.322 | 0.851 | 1.17x |
| `mul` | `[1073741824]` | 150.732 | 305.667 | 0.493 | 2.03x |
| `mul` | `[1024, 1024, 1024]` | 149.349 | 305.230 | 0.489 | 2.04x |

Interpretation:

- Torch `silu_and_mul` latency is close to Torch `silu + mul`, confirming that the benchmark baseline is effectively two high-quality Torch CPU vector loops.
- Gems `silu` alone is only moderately slower at 1G elements, while Gems `mul` alone is about 2x slower. This points strongly at generic pointwise streaming/memory/vector loop quality, not only `exp` lowering.
- Gems fused `silu_and_mul` is faster than Gems `silu + mul` separately, so fusion helps, but not enough. The fused Triton CPU loop still loses to Torch's two mature loops by roughly 1.4-1.5x on the 1G official shapes.

### Full Comprehensive Sweep, Warmup 10 / Iter 10

Date: 2026-07-01

Log root:

`$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/full-silu-and-mul-comprehensive-omp32-w10i10-20260701-004930`

Common environment:

- `OMP_NUM_THREADS=32`
- node3, cores `456-487`
- current workspace import shim enabled
- `--mode operator --level comprehensive --warmup 10 --iter 10 --record log`

Executed cases:

- `FlagGems/benchmark/test_silu_and_mul.py`: passed, `1 passed in 1017.17s`.
- `FlagGems/benchmark/test_mul.py::test_mul`: passed, `1 passed in 1025.98s`.

Float32 `silu_and_mul` comprehensive results:

| Shape | Torch ms | Gems ms | Speedup | Gems/Torch |
| --- | ---: | ---: | ---: | ---: |
| `[1073741824]` | 409.071 | 536.350 | 0.763 | 1.31x |
| `[64, 64]` | 0.048 | 0.435 | 0.110 | 9.07x |
| `[4096, 4096]` | 5.150 | 7.321 | 0.703 | 1.42x |
| `[64, 512, 512]` | 5.197 | 7.289 | 0.713 | 1.40x |
| `[1024, 1024, 1024]` | 354.787 | 497.944 | 0.713 | 1.40x |
| `[268435456]` | 60.976 | 107.159 | 0.569 | 1.76x |
| `[10000, 1]` | 0.072 | 0.579 | 0.124 | 8.07x |
| `[10000, 256]` | 0.837 | 1.935 | 0.433 | 2.31x |
| `[10000, 65536]` | 196.681 | 304.791 | 0.645 | 1.55x |
| `[100, 1, 100]` | 0.059 | 0.401 | 0.146 | 6.84x |
| `[100, 256, 100]` | 0.785 | 1.796 | 0.437 | 2.29x |
| `[100, 65536, 100]` | 195.541 | 304.253 | 0.643 | 1.56x |

Float32 `mul` comprehensive results:

| Shape | Torch ms | Gems ms | Speedup | Gems/Torch |
| --- | ---: | ---: | ---: | ---: |
| `[1073741824]` | 167.919 | 401.597 | 0.418 | 2.39x |
| `[64, 64]` | 0.010 | 2.412 | 0.004 | 237.48x |
| `[4096, 4096]` | 2.540 | 4.462 | 0.569 | 1.76x |
| `[64, 512, 512]` | 2.586 | 4.455 | 0.580 | 1.72x |
| `[1024, 1024, 1024]` | 170.991 | 399.960 | 0.428 | 2.34x |
| `[1024, 1]` | 0.011 | 1.227 | 0.009 | 112.85x |
| `[1024, 16]` | 0.015 | 0.630 | 0.024 | 41.05x |
| `[1024, 256]` | 0.033 | 0.507 | 0.065 | 15.31x |
| `[1024, 4096]` | 0.610 | 2.659 | 0.229 | 4.36x |
| `[1024, 65536]` | 6.729 | 14.489 | 0.464 | 2.15x |
| `[64, 64, 1]` | 0.012 | 2.512 | 0.005 | 209.08x |
| `[64, 64, 16]` | 0.039 | 0.966 | 0.041 | 24.50x |
| `[64, 64, 256]` | 0.071 | 2.403 | 0.029 | 33.99x |
| `[64, 64, 4096]` | 2.681 | 9.901 | 0.271 | 3.69x |
| `[64, 64, 65536]` | 32.292 | 122.171 | 0.264 | 3.78x |

Cross-dtype observations:

- `silu_and_mul` bf16 still has the largest dtype-specific gap on 1G shapes: `[1073741824]` speedup `0.333`, `[1024^3]` speedup `0.321`.
- `mul` bf16 is much worse than f32 on the largest contiguous shapes: `[1073741824]` speedup `0.077`, `[1024^3]` speedup `0.173`.
- `mul` f32 comprehensive confirms the earlier component signal: even without `exp`, the generic pointwise path can be 2.3-2.4x slower than Torch on 1G shapes and much worse on small tensors.
- `silu_and_mul` f32 is less bad than `mul` f32 on 1G shapes because fusion saves one memory pass, but it still loses to Torch's two mature CPU loops by 1.3-1.6x across large comprehensive shapes.
- The full 10/10 sweep reinforces that the primary float32 repair should target generic pointwise CPU loop quality and launch/scheduling overhead, not only the fused `silu` math expression.

## Compile-Time Impact

Benchmark mode matters:

- In `FlagGems/benchmark/performance_utils.py`, operator/wrapper latency runs `Config.warm_up` calls before timing `Config.repetition` calls.
- Therefore the normal benchmark command with `--warmup 20 --iter 10` mostly reports steady-state execution latency, not cold JIT compile latency.
- Compile time still matters for end-to-end first use and for workloads that create many fresh specializations.

Cold-cache first-call timing on `[10000, 65536]` with `ones` inputs:

1. Gems f32:
   - cold cache, `warmup=0, iter=1`: about 16.36 s
   - same cache, new process first call: about 3.68 s
   - warmup steady-state: about 354.5 ms/iter

2. Gems bf16:
   - cold cache, `warmup=0, iter=1`: about 16.21 s
   - warmup steady-state: about 543.4 ms/iter

3. Torch first call:
   - f32: about 237 ms
   - bf16: about 235 ms

Interpretation:

- Gems has an additional cold JIT compile cost of roughly 12-16 s for this specialization, while Torch does not.
- This compile gap is real and should be tracked separately.
- It does not explain the reported benchmark speedup gap under `--warmup 20 --iter 10`, because that benchmark intentionally times after warmup.
- It does affect first-token/first-use scenarios, autotuning loops, and broad shape/dtype sweeps where many Triton CPU specializations are generated.

## Optimization Directions

1. Add a dedicated Kunpeng `CodeGenConfig`.
   - Do not keep relying on NVIDIA defaults.
   - Initial candidates should include at least:
     - default-safe: `max_tile_size=512`, `prefer_block_pointer=True`
     - f32 candidate: `max_tile_size=1024`, `prefer_block_pointer=True`
     - dense 1D candidate: `max_tile_size=128/512`, `prefer_1d_tile=True`
   - Avoid one global replacement until more ops/shapes are measured.

2. Make pointwise codegen adaptive.
   - Select tile strategy by dtype, op complexity, contiguity, and numel bucket.
   - A single global tile size is too brittle for Kunpeng.
   - Candidate buckets:
     - tiny tensors: reduce wrapper/launch overhead first.
     - large contiguous pointwise: use dense 1D or tuned block-pointer path.
     - expensive math ops (`exp`, `div`, `gelu`, `silu`): consider smaller tile to reduce register/vector pressure, but account for more grid-stride iterations.

3. Add a dense contiguous 1D fast path for CPU/Kunpeng.
   - For `all_the_same_shape && all_c_contiguous`, generate pointer+mask dense indexing directly.
   - Benchmark against block pointer path per dtype and op family.
   - Keep fallback paths for strided/broadcasted cases.

4. Revisit CPU grid/CTA policy.
   - `max_grid_size=(65536, 65536, 65536)` is GPU-style and not clearly matched to Kunpeng CPU scheduling.
   - Investigate a CPU-aware cap based on physical cores, runtime worker count, and loop chunking.
   - Tune `tiles_per_cta` so very large tensors avoid either too many small inner loops or overly large vector tiles.

5. Treat bf16 as a separate lowering repair item.
   - Current TTIR/TTSharedIR shows bf16 loads extended to f32, math performed in f32, then truncated back to bf16.
   - The machine supports bf16/SVE bf16 features, so verify whether the lowering path unnecessarily expands/emulates bf16.
   - Repair target:
     - reduce avoidable `bf16 -> f32 -> bf16` traffic/conversions,
     - avoid generic unsupported-float emulation where native lowering is available,
     - verify generated LLVM/SVE for bf16 pointwise math.

6. Use paired benchmark methodology for tuning.
   - Large shape results showed substantial run-to-run variance.
   - For config changes, use same-node paired runs and reverse-order rechecks.
   - Record generated wrapper, TTIR/TTSharedIR, `num_warps`, tile size, and `tiles_per_cta` beside latency.

## Dense 1D Fast Path Experiment

Date: 2026-06-29

Patch under test:

- File: `FlagGems/src/flag_gems/utils/pointwise_dynamic.py`
- Adds an optional CPU rank-1 dense contiguous kernel generated beside the existing block-pointer kernel.
- Wrapper condition:
  `in*_stride[0] == 1 && out*_stride[0] == 1`
- Dense kernel uses direct `ptr + offsets` loads/stores instead of `tl.make_block_ptr`.
- Strided/non-dense cases fall back to the original generated kernel.

Important environment note:

- The recorded default environment imported editable `flag_gems` from
  `$AGENT_DIR/test/triton-cpu/FlagGems/src`, not the active product worktree.
- To test this workspace patch, the benchmark runner removed the scikit-build editable finder and inserted `$TRITON_REPO_DIR/FlagGems/src` before running `pytest.main`.
- Generated code confirmed the new branch:
  - wrapper contains `use_dense_1d`
  - Triton cache contains `silu_and_mul_kernel_kernel_rank_1_dense_1d.*`

Paired official core float32 benchmark, node3/cores `456-487`, `warmup=3`, `iter=3`:

Baseline command used the installed/editable old implementation.
Patch command used current workspace source with the editable finder removed.

| Shape | Baseline Gems ms | Dense fast path Gems ms | Result |
| --- | ---: | ---: | --- |
| `[1073741824]` | 492.075 | 497.656 | no improvement |
| `[64, 64]` | 0.489 | 0.493 | no improvement |
| `[4096, 4096]` | 7.037 | 6.541 | about 7% faster |
| `[64, 512, 512]` | 11.696 | 6.553 | about 44% faster |
| `[1024, 1024, 1024]` | 492.355 | 493.677 | no improvement |

Interpretation:

- Direct pointer+offset codegen helps some one-tile-per-CTA contiguous shapes, especially `[64, 512, 512]` in this paired run.
- It does not solve the two worst 1G float32 shapes. Those shapes still use the grid-stride-loop path with many tiles per CTA, and the dominant cost remains the large vector math/body and scheduling structure.
- Therefore, "remove block pointer for dense rank-1" is not sufficient as the primary float32 large-shape fix.
- Next repair target should be CPU-aware grid/tile policy for the 1G grid-stride-loop path, or reducing the generated vector body/register pressure. This aligns with prior perf evidence that the hot time is inside the generated kernel, not wrapper dispatch.

## Perf Profiling Plan

1. Profile current default baseline first, without CodeGenConfig injection.
2. Use a standalone script that runs only `flag_gems.silu_and_mul`, not the Torch reference, so samples belong to Gems/Triton CPU execution.
3. Start with `[10000, 65536]` and dtype `float32`/`bfloat16`.
4. Collect:
   - `perf stat` for cycles/instructions/cache/branch/task-clock context.
   - `perf record` + `perf report` for symbol-level hotspots.
5. If symbols are too coarse, inspect generated `.so`/LLVM artifacts and rerun with call graph if allowed by kernel perf settings.

## 2026-06-30 Follow-Up Experiments

Branch: `codex/silu-pointwise-bandwidth-vector-body`

Implementation attempts tried on this branch:

1. Added Kunpeng-specific `mul` and `silu_and_mul` override modules so configs can be tested without changing generic FlagGems behavior. This source change has been rolled back.
2. Added `TRITON_SHARED_VECTOR_TRANSFER_FULL_UNROLL` in `triton-shared/backend/compiler.py`, defaulting to the old behavior (`1`/true), to test vector-transfer lowering without hard-changing the global pipeline. This source change has been rolled back.
3. Added `scripts/run_silu_pointwise_full_test.sh` to run float32 `silu_and_mul` and `mul` core benchmarks with explicit import-path reporting. This local helper is retained.

Important test-environment note:

- The recorded environment imported default `triton` and `flag_gems` from
  `$AGENT_DIR/test/triton-cpu`.
- To test the current branch, the benchmark shim must remove `ScikitBuildRedirectingFinder` and prepend:
  - `$TRITON_REPO_DIR/python`
  - `$TRITON_REPO_DIR/FlagGems/src`
- Without this, changes in this branch are not actually measured.

Results from quick screening:

| Attempt | Command scope | Key result | Interpretation |
| --- | --- | --- | --- |
| `mul` `max_tile_size=1024`, block pointer | `test_mul.py::test_mul`, core f32, warmup=1/iter=1 | 1G Gems about 596/355 ms | Larger tile did not improve the bandwidth path. |
| `mul` `max_tile_size=512`, `prefer_1d_tile=True` | large shape file f32, warmup=3/iter=3 | `[1073741824]` 493 ms, `[1024^3]` 314 ms, `[10000,65536]` 217 ms, `[100,65536,100]` 180 ms | Regular pointer/1d-tile path is not a clear win; PtrAnalysis reports unrewritten pointer loads/stores. |
| `silu_and_mul` `max_tile_size=256`, block pointer | core f32, warmup=1/iter=1 | 1G Gems about 784/1041 ms | Smaller tile reduced vector width but increased grid-stride-loop work enough to regress large shapes. |
| `TRITON_SHARED_VECTOR_TRANSFER_FULL_UNROLL=0` | large shape file f32, warmup=1/iter=1 | `[1073741824]` 1001 ms, `[1024^3]` 865 ms, `[10000,65536]` 804 ms, `[100,65536,100]` 336 ms | Disabling transfer-to-scf full unroll is not a good default; it does not solve the `<512xf32>` math-body pressure. |

Controlled OMP A/B after source rollback:

- Log root: `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/omp-ab-f32-w10i10-20260630-185157`.
- Scope: official benchmark core shapes only, `float32`, `warmup=10`, `iter=10`, node3 cores `456-487`, current source import shim enabled.
- Confirmed imports after rollback:
  - `flag_gems.mul: flag_gems.ops.mul`
  - `flag_gems.silu_and_mul: flag_gems.fused.silu_and_mul`
- All four cases passed: `silu_and_mul_omp32`, `mul_omp32`, `silu_and_mul_ompunset`, `mul_ompunset`.

`silu_and_mul` A/B:

| Shape | Torch OMP=32 ms | Torch unset ms | Gems OMP=32 ms | Gems unset ms | Gems delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[1073741824]` | 331.653857 | 367.967868 | 493.032551 | 497.358060 | -0.87% |
| `[64, 64]` | 0.054550 | 0.048447 | 0.571060 | 0.580525 | -1.63% |
| `[4096, 4096]` | 7.647014 | 9.744930 | 7.085133 | 13.583040 | -47.84% |
| `[64, 512, 512]` | 4.398322 | 10.038829 | 7.084203 | 7.035136 | +0.70% |
| `[1024, 1024, 1024]` | 331.471229 | 367.078876 | 489.833450 | 496.406531 | -1.32% |

`mul` A/B:

| Shape | Torch OMP=32 ms | Torch unset ms | Gems OMP=32 ms | Gems unset ms | Gems delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[1073741824]` | 148.024940 | 150.189233 | 299.347496 | 299.049091 | +0.10% |
| `[64, 64]` | 0.009251 | 0.010586 | 0.389600 | 0.449133 | -13.26% |
| `[4096, 4096]` | 1.818657 | 4.116130 | 4.281068 | 4.208755 | +1.72% |
| `[64, 512, 512]` | 1.828361 | 3.960896 | 4.280066 | 4.197645 | +1.96% |
| `[1024, 1024, 1024]` | 148.033094 | 149.560404 | 298.358154 | 301.273155 | -0.97% |

OMP conclusion:

- Explicit `OMP_NUM_THREADS=32` is important for reproducible Torch baseline numbers, and it can materially change benchmark speedup ratios because Torch latency moves a lot on some shapes.
- It is not the main reason for the large 1G Gems improvement: Gems latency changes only about `0.9%-1.3%` for `silu_and_mul` 1G/`1024^3`, and about `-0.1%-1.0%` for `mul` 1G/`1024^3`.
- The only large Gems-side OMP win in this A/B is `silu_and_mul [4096,4096]` (`13.58 ms` unset to `7.09 ms` with OMP=32). The same element count in `[64,512,512]` does not improve, so treat this as shape/runtime scheduling sensitivity rather than a confirmed compiler fix.
- Future reports should pin `OMP_NUM_THREADS=32` and compare both absolute Gems latency and Torch latency. Speedup alone is not reliable enough for attribution.

Fixed-OMP source variant retest:

- Correction to decision basis: `OMP_NUM_THREADS=32` is a fixed benchmark parameter, not an optimization variable.
- Baseline log root: `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-baseline-omp32-w10i10-20260630-191746`.
- Fixed scope for all source variants: official core shapes, `float32`, `OMP_NUM_THREADS=32`, `warmup=10`, `iter=10`, node3 cores `456-487`, current source import shim enabled.
- Baseline Gems latency:
  - `silu_and_mul`: `[1073741824]` 496.038 ms, `[64,64]` 0.451 ms, `[4096,4096]` 6.950 ms, `[64,512,512]` 6.945 ms, `[1024,1024,1024]` 495.155 ms.
  - `mul`: `[1073741824]` 297.752 ms, `[64,64]` 0.315 ms, `[4096,4096]` 4.212 ms, `[64,512,512]` 4.220 ms, `[1024,1024,1024]` 296.538 ms.

Source variant results, relative to the fixed OMP=32 baseline:

| Variant | Logs | Main result | Keep? |
| --- | --- | --- | --- |
| `mul` tile512 `prefer_1d_tile=True` | `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-mul-1dtile512-omp32-w10i10-20260630-192406` | 1G `+2.07%`, `1024^3 +2.29%`; mid shapes within noise | No |
| `mul` tile1024 block-pointer | `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-mul-bptr1024-omp32-w10i10-20260630-192736` | 1G `+17.17%`, mid shapes `+41.61%/+43.67%`, `1024^3 +18.04%` | No |
| `silu_and_mul` tile128 block-pointer | `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-silu-bptr128-omp32-w10i10-20260630-193227` | Mid shapes improve `-26.37%/-26.45%`, but 1G and `1024^3` regress `+2.20%/+1.90%` | Not globally |
| `silu_and_mul` tile256 block-pointer | `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-silu-bptr256-omp32-w10i10-20260630-193551`, rerun `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-silu-bptr256-rerun-omp32-w10i10-20260630-193926` | Results are unstable: first run has `1024^3 +108.71%`, rerun has 1G `+63.63%` and mid-shape regressions | No |
| `transfer_to_scf full_unroll=false` | `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-fullunroll-false-omp32-w10i10-20260630-194413` | Mostly noise/small mixed effects; `mul 1024^3 +2.60%` regresses | No |
| Shape-aware tile128 with duplicated default kernel | `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-final-shapeaware-tile128-omp32-w10i10-20260630-195112`, rerun `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-final-shapeaware-tile128-rerun-omp32-w10i10-20260630-195905` | Unstable and often very bad: run1 `silu_and_mul` 1G `+138.33%`; rerun small shape `+2913%` | No |
| Shape-aware tile128 importing original default kernel | `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-shapeaware-importdefault-tile128-omp32-w10i10-20260630-200313` | `[4096,4096] -23.91%`, but 1G `+51.07%`, `[64,64] +412.95%`, `[64,512,512] +22.42%`, `1024^3 +48.06%` | No |

### Warmup 100 / Iter 100 Targeted Retests

Date: 2026-07-01

Common scope:

- official core shapes only
- `float32`
- `OMP_NUM_THREADS=32`
- node3, cores `456-487`
- current workspace import shim enabled
- `--mode operator --level core --warmup 100 --iter 100 --record log`

Default `silu_and_mul` standalone rerun:

- Log root: `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/baseline-silu-only-rerun-omp32-w100i100-20260701-015858`
- Import: `flag_gems.silu_and_mul: flag_gems.fused.silu_and_mul`
- Status: passed

| Shape | Torch ms | Gems ms | Speedup |
| --- | ---: | ---: | ---: |
| `[1073741824]` | 448.794 | 584.200 | 0.768 |
| `[64,64]` | 0.032 | 0.775 | 0.041 |
| `[4096,4096]` | 9.714 | 10.435 | 0.931 |
| `[64,512,512]` | 9.102 | 13.870 | 0.656 |
| `[1024,1024,1024]` | 619.693 | 502.073 | 1.234 |

`silu_and_mul` tile128 block-pointer retest:

- Log root: `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-silu-bptr128-omp32-w100i100-20260701-014618`
- Temporary import: `flag_gems.silu_and_mul: _kunpeng.fused.silu_and_mul`
- Config: `CodeGenConfig(max_tile_size=128, max_grid_size=(65536,65536,65536), prefer_block_pointer=True, prefer_1d_tile=False)`
- Status: passed, source change rolled back

| Shape | Default Gems ms | Tile128 Gems ms | Result |
| --- | ---: | ---: | --- |
| `[1073741824]` | 584.200 | 786.704 | regresses about 34.7% |
| `[64,64]` | 0.775 | 1.241 | regresses about 60.1% |
| `[4096,4096]` | 10.435 | 10.841 | regresses about 3.9% |
| `[64,512,512]` | 13.870 | 10.836 | improves about 21.9% |
| `[1024,1024,1024]` | 502.073 | 624.169 | regresses about 24.3% |

Tile128 conclusion:

- `max_tile_size=128` does reduce the vector body width and can improve one mid-size shape, but it is not a global fix.
- The 100/100 run confirms the 10/10 signal: reducing tile size multiplies grid-stride-loop work enough to hurt the largest shapes.
- This direction should not be kept as a Kunpeng default. The better version of this idea is still lower-level arithmetic-vector splitting, where math lowering is capped without changing the outer pointwise tile/grid policy.

Default `mul` standalone reruns:

- Log roots:
  - `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/baseline-mul-only-omp32-w100i100-20260701-004037`
  - `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/baseline-mul-only-rerun-omp32-w100i100-20260701-012840`
- Import: `flag_gems.mul: flag_gems.ops.mul`
- Status: both passed

| Shape | Default run1 Gems ms | Default run2 Gems ms | Observation |
| --- | ---: | ---: | --- |
| `[1073741824]` | 297.452 | 396.247 | high run-to-run variance |
| `[64,64]` | 0.296 | 0.615 | high small-shape variance |
| `[4096,4096]` | 4.159 | 4.783 | mostly stable |
| `[64,512,512]` | 4.149 | 10.339 | high run-to-run variance |
| `[1024,1024,1024]` | 642.842 | 354.285 | very high run-to-run variance |

`mul` grid32768 retests:

- Log roots:
  - `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-mul-grid32768-omp32-w100i100-20260701-003238`
  - `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-mul-grid32768-rerun-omp32-w100i100-20260701-013727`
- Temporary import: `flag_gems.mul: _kunpeng.ops.mul`
- Config: `CodeGenConfig(max_tile_size=512, max_grid_size=(32768,65536,65536), prefer_block_pointer=True, prefer_1d_tile=False)`
- Status: both passed, source change rolled back

| Shape | Grid run1 Gems ms | Grid run2 Gems ms | Observation |
| --- | ---: | ---: | --- |
| `[1073741824]` | 318.796 | 323.175 | more stable than default, but not always faster than default run1 |
| `[64,64]` | 0.282 | 0.318 | stable |
| `[4096,4096]` | 4.129 | 4.235 | stable |
| `[64,512,512]` | 4.142 | 4.194 | stable |
| `[1024,1024,1024]` | 316.839 | 454.173 | still unstable |

Grid/CTA conclusion:

- Reducing `max_grid_size[0]` from `65536` to `32768` can reduce scheduling variance for some `mul` shapes, and the two grid runs are steadier on the 16M-element mid shapes.
- It is not a proven global improvement: `[1073741824]` is slower than the best default run, and `[1024^3]` still swings from 316 ms to 454 ms.
- This should not be kept as a standalone global `CodeGenConfig` change. A useful CPU grid policy likely needs to use worker count, numel bucket, and op cost together, not a fixed halved grid cap.
- The variance itself is an optimization clue: CPU pointwise benchmarking should collect paired/repeated runs for 1G shapes, and conclusions should be based on absolute Gems latency distributions rather than one speedup line.

Keep/drop decision after fixed-OMP retest:

- No source optimization from this batch is retained.
- The only remaining tile128 signal is `silu_and_mul [64,512,512]`, but it is not safe as a global change because 1G shapes regress substantially in the 100/100 retest and `[4096,4096]` is no longer a confirmed win.
- The attempted shape-aware wrappers are not safe either; adding a second pointwise wrapper or importing the original default wrapper caused unstable or severe regressions.
- The next viable direction should be lower-level vector-body splitting or a pointwise_dynamic/codegen-level CPU policy that accounts for numel, worker count, op cost, and dtype without adding multiple competing Python-level wrappers for the same fused op.

Rollback decision:

- Rolled back the Kunpeng `mul` override and restored `_kunpeng/ops/__init__.py` to export only `gelu`.
- Rolled back the Kunpeng `silu_and_mul` fused override and removed the generated `fused` package files.
- Rolled back the `TRITON_SHARED_VECTOR_TRANSFER_FULL_UNROLL` compiler switch and restored `vector.ApplyTransferToScfPatternsOp(full_unroll=True)`.
- Removed ignored pycache left by the temporary override modules.
- Retained only the analysis document and the local benchmark helper script.

IR observations:

- `mul` 1d-tile still lowers to `tensor<512xf32>` / `<512 x float>` bodies and emits partial-copy alloc/subview code for masked tails.
- Fused `silu_and_mul` with tile 512 still contains very large `<512 x float>` shuffle sequences and vector math declarations in LLVM IR.
- Tile 256 changes the body to `<256 x float>`, but performance worsens on 1G shapes because each CTA executes more grid-stride-loop iterations.
- Therefore, reducing vector width by simply lowering pointwise tile size is too blunt. The better target is splitting/lowering the expensive vector math body or improving native SVE-sized lowering while keeping grid-stride overhead controlled.

SVE/vector-lowering inspection:

- Current CPU feature detection reports AArch64 with `sve`/`sve2` and `sve_vscale=4`.
- Pointwise payloads use `_sve_transform` because this machine does not advertise
  `sme` and does advertise `sve`.
- The default SVE path is already active for this workload, so pipeline
  selection is not a separate optimization experiment.
- The current SVE transform already has a `transform.legalize(vscale=...)` step that calls `populateVectorUnrollPatterns`.
- However, the C++ `LegalizeOp::getShape` matcher only covers `arith::MulFOp`, `arith::AddFOp`, `arith::SelectOp`, `arith::CmpFOp`, vector transfer/transpose, and contractions. It does not cover `math.exp` or `arith.divf`, which are the expensive parts of `silu_and_mul`.
- The pipeline order also matters: the normal SVE path runs `legalize` before `test-math-polynomial-approximation`, so the arithmetic produced from `exp` polynomial lowering is not re-legalized by default.
- The current transform pipeline exposes transfer-related vector patterns such as `ApplySplitTransferFullPartialPatternsOp`, `ApplyLowerTransferPatternsOp`, and `ApplyTransferToScfPatternsOp(full_unroll=True)`.
- Introspecting the current Python transform bindings also shows no arithmetic vector unroll/split op; available vector transform ops are transfer/mask/broadcast/transpose/contract/lowering patterns.
- The earlier `full_unroll=false` experiment only changes transfer-to-SCF lowering; it is not the missing arithmetic-vector split.

Second-legalize experiment:

- Temporary patch under test: add `TRITON_SHARED_SVE_LEGALIZE_AFTER_MATH=1` to run `transform.legalize(vscale=...)` immediately after `test-math-polynomial-approximation` and before `tptr-to-llvm`.
- Smoke log: `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-sve-relegalize-smoke-20260701-022004`
- 100/100 log: `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-sve-relegalize-omp32-w100i100-20260701-022309`
- 100/100 rerun log: `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-sve-relegalize-rerun-omp32-w100i100-20260701-023604`
- All runs passed and source patch was rolled back.

| Shape | Default Gems ms | Re-legalize run1 Gems ms | Re-legalize run2 Gems ms | Conclusion |
| --- | ---: | ---: | ---: | --- |
| `[1073741824]` | 584.200 | 500.589 | 1867.070 | unstable, cannot keep |
| `[64,64]` | 0.775 | 0.425 | 1.763 | unstable, cannot keep |
| `[4096,4096]` | 10.435 | 7.165 | 13.023 | unstable |
| `[64,512,512]` | 13.870 | 7.157 | 53.001 | unstable, severe regression in rerun |
| `[1024,1024,1024]` | 502.073 | 495.385 | 512.804 | roughly neutral |

Interpretation:

- Re-running legalize after polynomial lowering hits a plausible compiler point and can improve several shapes in one run, but it is too unstable to keep as a direct patch.
- A correct fix likely needs a targeted extension of `LegalizeOp::getShape` or a new transform pattern that explicitly covers `math.exp`/`arith.divf` and controls unroll target shape, instead of blindly re-running global legalize after math expansion.
- Because `LegalizeOp` lives in the external LLVM/MLIR tree, validating that targeted matcher change would require an LLVM/libtriton rebuild rather than only a Python-side patch in this workspace.

Updated optimization expectations:

1. `mul` bandwidth path:
   - Expected useful fix is not `max_tile_size=1024` or generic `prefer_1d_tile=True`.
   - More likely target: preserve block-pointer/tptr rewrite for dense rank-1 while reducing tail/partial-transfer overhead and CPU scheduling overhead.

2. `<512xf32>` vector body pressure:
   - Expected useful fix is not `transfer_to_scf full_unroll=false`.
   - More likely target: split expensive elementwise math (`exp`, `div`) into SVE/native-vector chunks or introduce a transform pattern that caps arithmetic vector body without multiplying grid-stride-loop iterations.

3. Shape policy:
   - `[10000,65536]` can benefit from smaller/vector-lighter variants in some earlier tests, but 1G shapes regress.
   - A single global config cannot satisfy all large float32 shapes; the repair needs op/shape/dtype-aware policy or a lower-level vector-body split.

### Warmup 10 / Iter 10 Three-Valid Retest

Date: 2026-07-01

Purpose:

- Re-test the representative optimization points with `warmup=10`, `iter=10`.
- Use only normal samples. A sample is valid only when:
  - `[1073741824]` and `[1024,1024,1024]` are close for both Torch and Gems (`<=10%` pair delta),
  - Torch large-shape mean stays in the expected normal range (`silu_and_mul: 300-430 ms`, `mul: 120-190 ms`).
- Average the first three valid samples for each variant.

Log root:

- `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/retest-optim-w10i10-omp32-20260701-avg3-c`
- Summary JSON: `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/retest-optim-w10i10-omp32-20260701-avg3-c/valid_average_summary.json`

Notes:

- One earlier `silu_bptr128` run failed before benchmarking because the initial stdin-defined monkeypatch kernel could not be inspected by `triton.jit`; it is not included in averages.
- The runner uses file-backed local variant kernels in `silu_pointwise_variants.py`.
- Temporary compiler changes for the `sve_relegalize` experiment were rolled back after the test; tracked source diff is clean.

`silu_and_mul` float32 operator/core averages:

| Variant | Valid samples | `[1073741824]` Gems ms | Delta vs baseline | `[1024,1024,1024]` Gems ms | Delta vs baseline | Mid-shape Gems ms (`[4096,4096]` / `[64,512,512]`) | Conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| baseline | 3/3 | 494.068 | 0.00% | 493.129 | 0.00% | 6.798 / 6.797 | Reference |
| tile128 block-pointer | 3/5 | 536.122 | +8.51% | 530.542 | +7.59% | 4.865 / 4.854 | Improves mid/small shapes, regresses 1G shapes; not global |
| tile256 block-pointer | 3/3 | 470.363 | -4.80% | 469.480 | -4.80% | 5.772 / 5.774 | Best signal in this 10/10 retest; promising but conflicts with earlier unstable 100/100 evidence |
| second legalize after math | 3/3 | 493.387 | -0.14% | 494.073 | +0.19% | 7.277 / 7.274 | Neutral on 1G, regresses mid shapes; do not keep |

`mul` float32 operator/core averages:

| Variant | Valid samples | `[1073741824]` Gems ms | Delta vs baseline | `[1024,1024,1024]` Gems ms | Delta vs baseline | Mid-shape Gems ms (`[4096,4096]` / `[64,512,512]`) | Conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| baseline | 3/3 | 297.444 | 0.00% | 297.938 | 0.00% | 4.164 / 4.153 | Reference |
| grid32768 | 3/3 | 301.542 | +1.38% | 302.456 | +1.52% | 4.191 / 4.188 | Slight regression; not a fix |
| 1d tile512 | 3/3 | 300.632 | +1.07% | 300.846 | +0.98% | 5.047 / 4.194 | Slight large-shape regression and mid-shape regression; not a fix |
| tile1024 block-pointer | 3/3 | 300.523 | +1.04% | 299.919 | +0.66% | 4.171 / 4.169 | Roughly neutral/slight regression; not a fix |

Interpretation:

- Under the stricter "three valid samples" rule, `silu_and_mul` tile256 is the only variant with a broad positive 10/10 signal, but it should be treated as a candidate needing 100/100 confirmation because earlier tile-size experiments were noisy.
- `silu_and_mul` tile128 is not a global fix: it improves mid-size and small shapes but loses about `8%` on both 1G shapes.
- The post-math second-legalize experiment no longer shows the dramatic swings seen in the earlier 100/100 reruns, but it also does not improve 1G and regresses mid shapes.
- The `mul` bandwidth-path variants remain negative or neutral; `mul` still needs a lower-level bandwidth/scheduling fix rather than a simple CodeGenConfig change.

## Direction Status After 100/100 Retests

| Direction | Status | Evidence | Conclusion |
| --- | --- | --- | --- |
| Dedicated Kunpeng `CodeGenConfig` | Tried | tile128/tile256/tile1024/1d-tile/grid32768 variants across 10/10 and selected 100/100 retests | Do not add one fixed global config now; performance is shape-sensitive and sometimes unstable. |
| `silu_and_mul` tile128 / smaller vector body | Tried at 100/100 | `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-silu-bptr128-omp32-w100i100-20260701-014618` vs default `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/baseline-silu-only-rerun-omp32-w100i100-20260701-015858` | Not a global fix; improves only `[64,512,512]`, regresses 1G and `[1024^3]`. |
| `silu_and_mul` tile256 block-pointer | Candidate from 10/10 three-valid retest | `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/retest-optim-w10i10-omp32-20260701-avg3-c`, valid runs 1/2/3 | Best 10/10 signal: about `-4.8%` on both 1G shapes and `-15%` on mid shapes. Needs 100/100 confirmation before patching. |
| `mul` CPU grid/CTA cap | Tried at 100/100 | grid32768 logs `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-mul-grid32768-omp32-w100i100-20260701-003238` and `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-mul-grid32768-rerun-omp32-w100i100-20260701-013727` | Fixed `max_grid_size[0]=32768` is not enough; may reduce mid-shape variance but is not reliably faster on 1G shapes. |
| Dense contiguous 1D fast path | Tried | dense direct pointer experiment from 2026-06-29 | Helps some mid-size contiguous shapes, not the worst 1G float32 shapes; not primary repair. |
| Python-level shape-aware wrappers | Tried | shape-aware tile128 duplicate/import-default runs | Drop; multiple Python-level wrappers introduce severe instability and small-shape overhead. |
| `transfer_to_scf full_unroll=false` | Tried | `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-fullunroll-false-omp32-w10i10-20260630-194413` | Drop; does not reduce the relevant `<512xf32>` math-body pressure and can regress. |
| Lower-level arithmetic-vector splitting | Tried via second `legalize` at 100/100; do not keep | `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-sve-relegalize-omp32-w100i100-20260701-022309` and rerun `$AGENT_DIR/logs/AgentForTritonCPU/legacy-agent-local/logs/variant-sve-relegalize-rerun-omp32-w100i100-20260701-023604`; `LegalizeOp` currently misses `math.exp`/`arith.divf` | Correct repair point identified, but blind re-legalize is unstable. Follow-up requires targeted `LegalizeOp`/transform matcher changes and likely LLVM/libtriton rebuild. |
| bf16 lowering repair | Out of current float32 baseline, evidence collected | TTSharedIR/LLVM/perf show bf16 emulation and shuffle/spill overhead | Track as separate repair point; do not mix it with the float32 performance conclusion. |
| Benchmark methodology | Adopted | OMP fixed, 100/100 baseline and repeated 100/100 targeted retests | Keep `OMP_NUM_THREADS=32`; require repeated paired runs for noisy 1G shapes. |

Background-test note:

- The skill contains `scripts/run_silu_pointwise_full_test.sh`.
- Detached background processes started from this tool are cleaned up unless hosted externally; `screen` also does not survive the tool process cleanup for long benchmark runs.
- Run the script directly from a user terminal when a true unattended long run is needed:
  `LOG_ROOT="$AGENT_DIR/logs/AgentForTritonCPU/silu-pointwise-manual" OMP_NUM_THREADS=32 WARMUP=100 ITER=100 LEVEL=core USE_CURRENT_TRITON=1 FULL_UNROLL=1 bash "$AGENT_DIR/AgentForTritonCPU/skills/silu-pointwise/scripts/run_silu_pointwise_full_test.sh"`
