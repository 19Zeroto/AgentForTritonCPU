---
name: triton-cpu-rebuild-test
description: Rebuild the local LLVM/MLIR install and Triton CPU editable package after compiler changes, then run the dgeglu float32 FlagGems benchmark with reproducible external logs.
---

# Triton CPU Rebuild Test

Use this skill after changing Triton CPU, TritonShared, MLIR lowering, or
compiler-related source. It covers the current dgeglu validation path:
update LLVM, rebuild/install LLVM, clean Triton's generated Python build tree,
reinstall Triton CPU, and run the `dgeglu` benchmark with `float32`.

## Entry point

Run the wrapper from any directory:

```bash
OMP_NUM_THREADS=32 \
numactl --physcpubind=288-319 --membind=2 \
bash "$AGENT_DIR/AgentForTritonCPU/skills/triton-cpu-rebuild-test/scripts/rebuild_and_test.sh" \
  --cpu-node 2 --mem-node 2 --cpu-list 288-319
```

The default is 32 OpenMP threads, `operator` mode, `core` level, five warmup
iterations, and five measured iterations. Set NUMA and CPU arguments to match
the machine; the example binding is not universal.

The outer `numactl` is mandatory for the complete workflow, not only for the
benchmark. Every child command—including environment setup, `git`, `ninja`,
build cleanup, `pip install`, import checks, and the benchmark—inherits CPU
288-319 and memory NUMA node 2 binding. Keep the outer binding and the runner's
`--cpu-node`/`--mem-node`/`--cpu-list` consistent.

Inspect the generated run directory for LLVM/Triton build logs, the import
check, benchmark stdout, the benchmark manifest, and the GBPS report. Results
are written below `$AGENT_DIR/logs/AgentForTritonCPU/triton-cpu-rebuild-test/`,
outside the source repositories.

## Workflow constraints

- The script sources the shared environment helper, which activates
  `triton-yzc` when available and verifies LLVM/MLIR and Triton paths.
- By default it runs `git pull --ff-only` in `llvm-project`, then `ninja` and
  `ninja install` in `llvm-project/build`.
- Before reinstalling Triton CPU it removes only
  `triton-cpu/python/build`, preventing stale generated extensions from being
  reused. The exact command and path are recorded in the run log.
- The Triton install is
  `python3 -m pip install --no-build-isolation -e python` from the repository
  root. Use `--skip-llvm-update` or `--skip-install` only when intentionally
  reusing an already prepared build.
- The complete script invocation must be started under the same outer
  `numactl` binding as the benchmark. Do not run environment setup, LLVM
  build/install, cleanup, Triton installation, import checks, or benchmark as
  unbound standalone commands.
- The benchmark is delegated to the existing fusion-metrics runner, so dgeglu's
  benchmark-local `gbps` calculation and report format remain the source of
  truth.
- `float32` is fixed by the wrapper; it does not silently fall back to another
  dtype.
- `--dry-run` prints all commands without pulling, building, deleting, testing,
  or creating result files. The wrapper never commits or pushes changes.

Read [references/rebuild-test.md](references/rebuild-test.md) when changing the
workflow, output layout, or machine-binding defaults. Execute
`scripts/rebuild_and_test.sh --help` before adapting the command.
