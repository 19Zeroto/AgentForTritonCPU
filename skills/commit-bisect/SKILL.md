---
name: commit-bisect
description: Locate the first Triton CPU commit that breaks a reproducible pytest target by rebuilding candidate commits, classifying test results for git bisect, preserving logs, and restoring the original checkout. Use when a known-good and known-bad revision bracket a Triton CPU regression.
---

# Bisect Triton CPU Regressions

## Required inputs

- Triton CPU repository path.
- Known-good commit.
- Known-bad commit or `HEAD`.
- Reproducible pytest file or node ID.

## Workflow

1. Inspect repository status. Stop when tracked changes exist; never hide or
   discard them.
2. Load the `environment` skill and confirm LLVM/MLIR paths.
3. Review options:

   ```bash
   bash scripts/bisect_triton_commit.sh --help
   ```

4. Run with `--good`, `--bad`, `--test`, and explicit overrides when needed.
5. Treat build, install, collection, or environment failures as skipped commits,
   not good/bad evidence.
6. Confirm original checkout restoration and report first-bad commit plus log
   directory.

## Resources

- Execute `scripts/bisect_triton_commit.sh`.
- Read `references/design.md` before changing checkout restoration, probe
  classification, build steps, or log layout.

