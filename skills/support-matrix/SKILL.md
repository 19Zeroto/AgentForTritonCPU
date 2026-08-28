---
name: support-matrix
description: Generate evidence-backed Triton language and FlagGems operator support matrices by statically analyzing implementations, test mappings, dtypes, shapes, reductions, and call-site evidence. Use when updating operator coverage reports, auditing test support, or producing support CSV and Markdown artifacts.
---

# Generate Operator Support Matrices

## Workflow

1. Set `TRITON_REPO_DIR` when the product repository is not the workspace
   sibling.
2. Set `OUT_DIR` to an external writable directory; default is `/tmp`.
3. Choose generator:
   - Run `scripts/generate_triton_level_support_matrix.py` for `tl.*` call-site
     evidence and test mapping.
   - Run `scripts/generate_flaggems_operator_support_matrix.py` for FlagGems API,
     implementation, and test coverage.
4. Inspect generated matrix and evidence files together. Do not infer support
   from API input when the Triton operand was cast or reshaped.
5. Report output paths, row counts, covered rows, and unresolved mappings.

## Resources

- Execute generators under `scripts/`; outputs are deterministic static-analysis
  artifacts.
- Read `references/design.md` before changing evidence boundaries, dtype/shape
  semantics, matching rules, or output columns.

