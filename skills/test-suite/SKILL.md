---
name: test-suite
description: Run, resume, and summarize Triton CPU FlagGems correctness tests with marker-level or file-level parallelism, controlled xdist workers, external state files, cache cleanup, and full-suite support. Use for FlagGems test execution, batch validation, resumable test runs, or result summaries.
---

# Run FlagGems Test Suites

## Select an entrypoint

- Use `scripts/run_triton_tests.py` for `(file, marker)` work items, marker
  filtering, cached collection, and resume.
- Use `scripts/run_simple_tests.py` for simpler file-level execution and
  per-file Triton cache cleanup.
- Use `scripts/summarize_results.py` to read marker-runner state.
- Use `scripts/run_all_flaggems_tests.sh` for direct all-file pytest batches.
- Use `scripts/run_flaggems_full.sh` for the full-suite wrapper.

## Workflow

1. Read `../../agent/rules/testing.md` and `../../agent/playbooks/testing.md`.
2. Load the `environment` skill before running tests.
3. Follow the test method defined by the testing playbook. Confirm
   `OMP_NUM_THREADS <= 32` and xdist workers `<= 32`.
4. Run the smallest relevant entrypoint. Start with `--help` when parameters are
   unclear.
5. Keep default state and logs under
   `$AGENT_DIR/logs/AgentForTritonCPU/test-suite/`, or pass explicit paths.
6. Summarize actual totals and first root-cause failure. Do not generalize a
   targeted pass to full-suite success.

## Resources

- Execute files under `scripts/` without reading their full source.
- Read `references/design.md` before changing scheduling, state format, resume
  behavior, logging, or cache cleanup.
