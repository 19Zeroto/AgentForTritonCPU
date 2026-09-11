# Documentation and skill cleanup

## Goal

Make the Agent control plane concise, internally consistent, and portable while
preserving the reusable Triton CPU workflows.

## Scope

- Repair workspace routing and remove references to capabilities that do not
  exist in this repository.
- Give rules, playbooks, shared references, skill entrypoints, and historical
  evidence distinct ownership.
- Use one workspace/path contract and move machine-specific CPU/NUMA settings
  out of normative defaults.
- Consolidate overlapping test-suite entrypoints.
- Separate generic LLVM/Triton rebuild work from the dgeglu validation recipe.
- Keep profiling documentation aligned with current source behavior and remove
  the obsolete fusion-metrics skill whose formulas are now implemented in the
  main repository benchmark methods.
- Keep a human-readable copy of the reorganized guidance in `docs/`; keep
  executable contracts and Agent-only context under `agent/` and `skills/`.

## Constraints

- Do not modify `triton-cpu` or its existing user changes.
- Do not commit or push.
- Keep generated logs, caches, and validation output outside source trees.
- Preserve benchmark-local `get_tflops()` and `get_gbps()` as the authoritative
  metric definitions; profiler formulas are explicitly labeled estimates.

## Validation

- All local Markdown links resolve.
- Every routed skill exists and passes skill structure validation.
- Changed Python files compile; changed Shell files pass `bash -n`.
- Main entrypoints support `--help` and destructive workflows support a dry run.
- No concrete user home path or machine-specific CPU/NUMA default remains in
  active rules, playbooks, skill entrypoints, or default configuration.

## Limitations

- Full correctness, benchmark, LLVM build, and Triton installation runs are out
  of scope because they are expensive and would touch the product/toolchain
  environment.
- The base Python lacks `PyYAML`; skill validation and benchmark smoke checks
  use the configured `triton-yzc` environment, which provides project
  dependencies.
