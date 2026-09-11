---
name: environment
description: Prepare and verify the Triton CPU development environment by activating the configured Python environment and exporting LLVM, MLIR, TritonShared, FlagGems, cache, and repository paths. Use before Triton CPU builds, tests, benchmarks, profiling, or scripts that require the local toolchain.
---

# Prepare Triton CPU Environment

## Workflow

1. Resolve the workspace/install root from `AGENT_DIR`, defaulting to
   `$HOME/agent`.
2. Override `AGENT_DIR` when the workspace/install root differs; override
   `VENV_DIR`, `LLVM_INSTALL_DIR`, or `TRITON_REPO_DIR` when individual paths
   differ from the workspace layout.
3. Source the helper in the current shell:

   ```bash
   source "$AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"
   ```

4. Verify printed `python`, `LLVM_INSTALL_DIR`, `TRITON_REPO_DIR`,
   `TRITON_CACHE_DIR`, and `TRITON_SHARED_OPT_PATH`.
5. Stop if LLVM/MLIR or Triton paths are missing. Do not guess substitute paths.

## Resources

- Execute `scripts/triton-cpu-env.sh`; never launch it as a child process when
  exported variables must remain available.
- Read `references/bootstrap.md` only when preparing a new machine or an LLVM
  install that does not yet exist.

## Completion checks

- Keep `TRITON_CACHE_DIR` outside source repositories.
- Report whether Conda or venv was activated.
- Report missing toolchain components without claiming environment readiness.
