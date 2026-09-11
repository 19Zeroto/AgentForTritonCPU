---
name: test-suite
description: Run, resume, and summarize Triton CPU FlagGems correctness tests at file or pytest-marker granularity. Use for batch or full-suite validation; use the testing playbook directly for one or two targeted cases.
---

# Run FlagGems Test Suites

## Choose the granularity

- `scripts/run_simple_tests.py`: file-level execution, including full-suite runs.
  It supports include/exclude filters, resume state, per-file cache cleanup, and
  optional complete logs.
- `scripts/run_triton_tests.py`: `(file, pytest marker)` execution when smaller
  resumable work items or marker filtering are needed.
- `scripts/summarize_results.py`: summarize either runner's state file.

Do not add another wrapper for “all tests”: invoking the file runner without an
include filter already selects the full suite.

## Workflow

1. Read `../../agent/rules/testing.md` and
   `../../agent/playbooks/testing.md`, then load the `environment` skill.
2. Run `--help` on the selected entrypoint. Set concurrency and
   `OMP_NUM_THREADS` explicitly, each at or below 32.
3. Keep state and logs outside source repositories. Defaults are below
   `$AGENT_DIR/logs/AgentForTritonCPU/test-suite/`; use explicit paths when runs
   must be isolated.
4. Use resume only with the same test root and selection filters. State version
   mismatches start a new empty state rather than guessing compatibility.
5. Report actual totals, failed files or markers, and the first root-cause
   failure. A targeted pass is not full-suite success.

The runners stay in the foreground. Process supervision for a long run belongs
to the caller's terminal or job system, not to this skill.
