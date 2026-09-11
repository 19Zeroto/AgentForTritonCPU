# Kunpeng `silu_and_mul` case study

> Historical, non-normative evidence from 2026-06-27 through 2026-07-01. Re-run
> measurements against the current checkout before using these conclusions.

## Scope

The study compared FlagGems `silu_and_mul` with the PyTorch expression
`mul(silu(x), y)` on Kunpeng, primarily with float32 and 32 threads. It also
used `mul`/`add` controls and inspected generated IR and perf samples.

## Durable findings

- The initial large gap was partly caused by a one-thread profiler setup.
  Thread counts and affinity must be captured with every comparison.
- Under matched 32-thread conditions, large float32 shapes still lagged the
  PyTorch baseline by roughly 3–4× in the early runs.
- Large-shape perf samples were dominated by the generated JIT kernel; fixed
  launch overhead was more important for tiny shapes.
- Generated code showed oversized vector math, spill/store pressure, and in
  some paths a temporary buffer plus `memrefCopy`.
- bf16 had additional extension, shuffle, emulation, and spill costs and should
  be investigated separately from the float32 path.

## Experiments and status

| Direction | Status at the end of the study |
| --- | --- |
| Fixed Kunpeng tile/grid configuration | Shape-sensitive; no global default justified. |
| tile128 | Helped some middle shapes but regressed the largest shapes. |
| tile256 block-pointer | Promising in short repeated runs; needed longer confirmation. |
| Dense contiguous 1D fast path | Helped some shapes, not the worst large cases. |
| Python-level shape-aware wrappers | Dropped because of instability and overhead. |
| Disable full unroll | Did not address the relevant vector math pressure. |
| Blind second legalization after math | Unstable; not suitable as a patch. |
| Targeted arithmetic-vector legalization | Plausible lower-level follow-up. |
| bf16 lowering | Separate follow-up, not part of the float32 conclusion. |

## Reuse guidance

Use this case study to choose probes, not to prescribe a patch. A new run must
record the current commits, benchmark shapes, input distribution, warmup and
iteration counts, pipeline, CPU/NUMA binding, thread counts, cache policy, and
repeat stability. Use `flaggems-kernel-perf` for current instruction-level
attribution.
