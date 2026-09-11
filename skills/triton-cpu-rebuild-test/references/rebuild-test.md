# Triton CPU rebuild and dgeglu validation

## Why this wrapper exists

The workspace already has separate helpers for loading the Triton CPU
environment and for running FlagGems profiling. It did not have one entry point
that updated LLVM, rebuilt the LLVM install, refreshed Triton's generated build
tree, and then ran the exact dgeglu benchmark used by the optimization work.
This wrapper composes those existing helpers; it does not duplicate benchmark
metric formulas or alter product-repository configuration.

## Stages

1. Resolve `AGENT_DIR`, `TRITON_REPO_DIR`, `LLVM_SOURCE_DIR`,
   `LLVM_BUILD_DIR`, and `LLVM_INSTALL_DIR` from options or environment.
2. Source `skills/environment/scripts/triton-cpu-env.sh`. The helper activates
   the configured `triton-yzc` Conda environment when available, exports
   LLVM/MLIR, TritonShared, FlagGems, and an external Triton cache directory,
   and stops if required toolchain paths are absent.
3. Capture the Triton and LLVM branch/commit/dirty-file state in the external
   run directory. Existing user changes are not stashed, reset, or overwritten.
   LLVM update stops on a dirty LLVM worktree rather than mixing local changes
   with a remote update.
4. Unless `--skip-llvm-update` is given, run the following in `LLVM_SOURCE_DIR`:

   ```bash
   git pull --ff-only
   ninja -C build
   ninja -C build install
   ```

   The actual configured `LLVM_BUILD_DIR` is used instead of assuming the
   caller's current directory.
5. Unless `--skip-install` is given, remove the exact directory
   `TRITON_REPO_DIR/python/build`, then run the editable install from the Triton
   repository root:

   ```bash
   python3 -m pip install --no-build-isolation -e python
   ```

   This is the stage that rebuilds generated Triton CPU/TritonShared extensions
   needed by compiler or MLIR source changes. `--skip-install` skips both the
   clean and the install. The shared environment helper is sourced again after
   installation so `TRITON_SHARED_OPT_PATH` and the external cache refer to the
   freshly recreated build.
6. Import `triton` and `torch` with the active Python interpreter and record
   their locations/versions. This catches an accidentally different Python
   environment before spending time on the benchmark.
7. Invoke the existing fusion benchmark runner with `--ops dgeglu --dtypes
   float32 --mode operator`. The runner records latency and the benchmark-local
   logical GBPS value, then exports `fused_non_compute_bandwidth.csv`.

## Mandatory CPU and NUMA binding

The entire rebuild-and-test workflow must run with the same binding as the
benchmark. Start the wrapper with an outer `numactl`; child processes inherit
the policy, including environment setup, LLVM `git`/`ninja` commands, build
cleanup, `pip install`, import checks, and the benchmark itself:

```bash
OMP_NUM_THREADS=32 \
numactl --physcpubind=288-319 --membind=2 \
bash "$AGENT_DIR/AgentForTritonCPU/skills/triton-cpu-rebuild-test/scripts/rebuild_and_test.sh" \
  --cpu-node 2 --mem-node 2 --cpu-list 288-319
```

Do not invoke the wrapper or any of its rebuild/test stages without this
binding. If another NUMA node or CPU range is selected, change the outer
`numactl` and the wrapper's `--cpu-node`, `--mem-node`, and `--cpu-list`
together. The number of CPUs in the list must match `OMP_NUM_THREADS`.

## Output layout

```text
<output-root>/<timestamp>/
├── command.txt
├── environment.txt
├── llvm-status-before.txt
├── llvm-pull.log
├── llvm-build.log
├── llvm-install.log
├── clean-triton-build.log
├── install.log
├── import-check.log
└── benchmark/
    └── <runner-timestamp>/
        ├── run.json
        ├── records/dgeglu.log
        ├── stdout/dgeglu.log
        └── report/fused_non_compute_bandwidth.csv
```

All outputs are outside `triton-cpu`. A nonzero LLVM, install, import, or
benchmark return code stops the workflow and is preserved in the corresponding
log. With `--skip-llvm-update`, the LLVM update/build logs are not created.

## Reproducibility

Keep the following fixed when comparing two runs: LLVM revision, Triton commit,
dtype (`float32`), benchmark mode and level, warmup/iteration counts,
`OMP_NUM_THREADS`, NUMA nodes, CPU list, shape file, and whether the editable
install was rebuilt. The standard test rules require `OMP_NUM_THREADS <= 32`;
the wrapper enforces this limit.

For a compiler change, use the normal command once for the new build. Use
`--skip-llvm-update` only when LLVM has already been updated and installed, and
use `--skip-install` only for additional measurements of the same installed
Triton build. Do not interpret a benchmark result as evidence that a changed
MLIR component was used unless the relevant build/install logs completed
successfully and the import check points at the intended editable repository.
