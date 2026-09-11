# Rebuild workflow contract

`scripts/rebuild.sh` operates on an already configured workspace. Initial LLVM
and Python environment installation belongs to the environment bootstrap guide.

## Stages

1. Resolve paths from explicit options or the workspace contract.
2. Source the shared environment helper.
3. Record LLVM and Triton worktree state in an external run directory.
4. Unless skipped, require a clean LLVM worktree, run `git pull --ff-only`, then
   `ninja` and `ninja install` in the configured build directory.
5. Unless skipped, remove exactly `$TRITON_REPO_DIR/python/build` and run
   `python3 -m pip install --no-build-isolation -e python` from the Triton root.
6. Source the environment helper again and record an import check for `triton`
   and `torch`.

The first nonzero stage stops the workflow. Logs already produced remain in the
run directory. The script does not choose correctness or performance tests.

## Safety invariants

- Never stash, reset, commit, or push either repository.
- Never pull a dirty LLVM worktree.
- Never broaden cleanup beyond the resolved `triton-cpu/python/build` path.
- `--dry-run` prints mutation commands without creating a run directory or
  performing the mutation.
- `--skip-llvm-update` and `--skip-triton-install` are explicit evidence gaps
  and must be reported.

## Output

Runs are stored below
`$AGENT_DIR/logs/AgentForTritonCPU/triton-cpu-rebuild/<timestamp>/` unless
overridden. They contain the invocation, environment, repository status, and a
log for each executed stage.
