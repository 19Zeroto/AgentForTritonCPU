# FlagGems fusion metrics design

## Architecture

`FlagGems/benchmark/fusion_metrics_formulas.yaml` is the fixed runtime source of
truth. Product function names without the `flag_gems.` prefix are primary keys.
`skip_layer_norm` and `weight_norm` declare the benchmark-name aliases
`skip_layernorm` and `weight_norm_interface`. Aliases are globally unique and
cannot conflict with primary keys.

`FlagGems/benchmark/fusion_metrics.py` owns YAML loading, formula and alias
lookup, input binding, primitive dispatch, evaluation, and formula formatting.
Keep it small: the fixed repository YAML is trusted input, so the evaluator
does not need broad schema validation, AST validation, or detailed validation
error construction. `performance_utils.py` retains only an opt-in call at the
point where benchmark args/kwargs are visible. Normal benchmarks neither load
the YAML nor execute fusion expressions.

The runner is Agent tooling. It loads the fixed config, discovers pytest
markers, chooses dtype/pipeline, runs isolated NUMA jobs, collects record logs,
computes CSV rows, and merges successful single-operator reruns. Formula
evaluation remains in `fusion_metrics.py`; the runner must not add
operator-specific calculation functions.

Compute report targets are selected from the runner's scaled `(pipeline,
dtype)` table. The scale is `0.2 / 304 * 32`: SME/float32 is 2.52631579
TFLOPS, SME/bfloat16 is 5.05263158 TFLOPS, and SVE/float32 is 0.71578947
TFLOPS. An explicit `--target-tflops` or `FUSION_TARGET_TFLOPS` value
overrides the table uniformly for calibration.

Memory report default target bandwidth is 30.72 GB/s; `--target-gbps` and
`FUSION_TARGET_GBPS` remain explicit overrides.

For an explicitly selected dtype, the runner retries each failed operator with
BF16 and then without `--dtypes`. Retry attempts are isolated per operator;
terminal output and coverage notes identify the fallback, while every attempt
keeps separate raw output and record-log paths. Implicit dtype selection keeps
the existing default behavior.

The runner optionally reads defaults from the fixed adjacent `scripts/.env`.
It parses only the documented `FUSION_*` whitelist as restricted `KEY=VALUE`
data: no sourcing, interpolation, command substitution, includes, `export`, or
arbitrary environment changes. CLI values override `.env`, and code defaults
remain active for absent or empty values. Configured paths must be absolute or
start with `~`. The manifest records only final effective values, never `.env`
content, location, or hash.

## Opt-in contract

The runner sets both variables for each pytest child:

- `FLAGGEMS_FUSION_METRICS_CONFIG`: absolute YAML path.
- `FLAGGEMS_FUSION_METRICS_KEY`: current primary formula key.

Both must be present. The benchmark `op_name` must equal the primary key or one
of its aliases. Each formula exposes exactly one primary value:
`logical_flops` for `compute`, or `logical_bytes` for `memory`.

## YAML schema

Keep the existing `version: 1` YAML structure unchanged. Each formula keeps:

- `category`: `compute` or `memory`.
- `metric`: respectively `logical_flops` or `logical_bytes`.
- `inputs`: named bindings using `position`, `keyword`, and optional `default`.
- `expression`: restricted expression text.
- `enabled`: boolean.

Optional `aliases` provide benchmark naming exceptions. Disabled formulas keep
their existing `exclude_reason`.

The evaluator binds actual benchmark args/kwargs using the existing
`position`, `keyword`, and `default` data, then evaluates the expression with
the bound values and required primitives. Keep only logic needed to produce one
numeric `logical_flops` or `logical_bytes` result. Do not add general-purpose
schema validation or an operator-specific evaluator.

## Formula display

Keep runtime binding names in YAML unchanged. A small display-name mapping
handles only names that differ from the public operator signature, such as
`input` to `A` for `silu_and_mul`, `input` to `x` for
`fused_add_rms_norm`, and `value`/`scale` to `v`/`g` for `weight_norm`.
This mapping affects documentation only.

Use one generic formula formatter. Expand common helpers directly:

- `nbytes(x)` to `x.numel() * x.element_size()`.
- `numel(x)` to `x.numel()`.
- `dim(x, i)` to `x.shape[i]`.
- `shape_numel(x)` to `product(x)`.
- `tensor_sum(x)` to `sum(x)`.
- `count_nonzero(x)` to `x.count_nonzero()`.
- `output_nbytes_like(x, n)` to `n * x.element_size()`.

Complex attention and sparse primitives may keep their runtime calculation
functions, but the formatter must expand them into the corresponding input
shape and summation formula. Do not create one formatter or calculator per
operator.

Current primitives cover tensor `numel`/logical bytes, derived output bytes,
shape dimensions/products, tensor sum/nonzero count, attention pair counts,
varlen attention, and sparse MLA. Add a primitive only when the expression
cannot remain clear and mechanically equivalent with existing operations.

## Metric conventions

One multiply-add counts as 2 FLOPs. Compute formulas count the existing primary
matrix work and intentionally omit secondary activation, softmax, scale, and
index work. Logical bytes count semantic input reads and output writes with
actual element sizes; implementation temporaries and hardware transactions are
excluded.

Keep migration mechanically equivalent. Do not silently correct formulas while
moving or refactoring them. These pass-count conventions remain audit items:

- `fused_add_rms_norm`: 2 passes below normalized size 4096, otherwise 3.
- `instance_norm`: 2 input passes below spatial size 4096, otherwise 3; running
  statistics use 2 passes with input stats and 1 otherwise.
- `skip_layer_norm`: input uses 2/3 passes at normalized size 4096; residual
  uses 1/3 passes.

`moe_align_block_size_triton`, `weight_norm`, and `topk_softmax` retain formulas
with `enabled: false`; coverage reports them as `SKIPPED`.

## Reports

Keep compute/memory CSV headers and row order stable. Append two notes: fixed
YAML location and current SHA-256. Preserve coverage, per-run raw output,
record-log, and merge structures. Default output is
`$AGENT_DIR/logs/fusion_metrics/`; historical product-repository results are
not migrated.

`--export-formulas OUTPUT.md` loads only the fixed YAML and must not invoke
runtime environment, NUMA, or pytest preflight. Preserve a Markdown table with
exactly four columns:

- `Product function`: configured operator name.
- `Category`: `compute` or `memory`.
- `Formula`: direct formula using original public input names.
- `Status`: `enabled` or `disabled: <exclude_reason>`.

Do not include aliases, input binding JSON, config paths, or SHA-256 metadata in
the Markdown export.

## Maintenance workflow

Before changing expressions, evaluator primitives, runner architecture, metric
scope, or pass-count conventions, review this document completely.

Keep one primary metric per formula. Bind only actual benchmark args/kwargs.
Represent derived outputs with whitelisted primitives such as
`output_nbytes_like`. Do not add arbitrary evaluation, formula edit CLIs, or
formula path flags.

For the current simplification, keep the YAML structure and existing result
collection unchanged. Limit implementation changes to
`FlagGems/benchmark/fusion_metrics.py` and the runner's formula export path. Do
not add test files, schema versions, compatibility layers, or new modules.

After implementation, export the Markdown once and confirm the four columns,
original input names, and expanded formulas. Run a representative operator only
when checking the runtime calculation path.
