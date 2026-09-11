---
name: triton-cpu-rebuild
description: Update and install an existing LLVM/MLIR build, clean Triton's generated Python build tree, reinstall Triton CPU, and record reproducible external logs. Use after compiler or lowering changes; optionally run the maintained dgeglu float32 validation recipe.
---

# Rebuild Triton CPU

This skill separates toolchain mutation from validation. Rebuilding proves that
the intended sources were installed; it does not by itself prove correctness or
performance.

## Generic rebuild

1. Inspect both the LLVM and Triton worktrees. The script refuses to pull a
   dirty LLVM checkout and never stashes, resets, commits, or pushes changes.
2. Review options and start with a dry run:

   ```bash
   bash scripts/rebuild.sh --help
   bash scripts/rebuild.sh --dry-run
   ```

3. Run without `--dry-run` only when LLVM update/build/install and removal of
   the exact `triton-cpu/python/build` directory are intended.
4. Read `references/workflow.md` before changing stage order, cleanup scope, or
   log layout.

## dgeglu validation recipe

Use this only when dgeglu float32 is a relevant acceptance signal. Select the
current machine's NUMA nodes explicitly:

```bash
bash scripts/rebuild_and_test_dgeglu.sh \
  --cpu-node <cpu-node> --mem-node <memory-node> \
  --cpu-list <cpu-list> --dry-run
```

Read `references/dgeglu-validation.md` before changing the composition or
benchmark defaults. For other validation targets, use the testing playbook or
the relevant performance skill instead of extending this recipe.

## Completion

- Report the run directory and the completed rebuild stages.
- Confirm the imported Triton path points to the intended checkout.
- Report validation separately, including skipped or unexecuted stages.
