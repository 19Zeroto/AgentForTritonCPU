# Triton / FlagGems Operator Coverage Reference

## Contents

- [Two Evidence Layers](#two-evidence-layers)
- [Evidence Chain](#evidence-chain)
- [Triton Matrix Contract](#current-triton-matrix-script-contract)
- [FlagGems Matrix Contract](#current-flaggems-matrix-script-contract)
- [Mapping Rules](#flaggems-test-mapping-rules)
- [Shape And Rank Rules](#shape-and-rank-rules)
- [Output CSV Contract](#current-output-csv-contract)
- [Rule Of Thumb](#rule-of-thumb)

This note records the coverage model for Triton language operators and FlagGems
operators. Keep the two evidence layers separate.

## Two Evidence Layers

### Triton-op evidence

Question: what does a Triton language op, such as `tl.sum`, actually receive?

Use the `tl.*` call site as the boundary. For each call, inspect the operand
right before the call:

- operand dtype passed into `tl.*`
- operand rank/tile shape passed into `tl.*`
- reduction axis / keep_dims where relevant
- conversions applied before the `tl.*` call

Do not copy the FlagGems API input dtype/shape directly into the Triton-op row.
If a FlagGems kernel loads `float16` and casts to `float32` before `tl.sum`, the
Triton `sum` evidence is `float32`, with a note that `float16` input was converted
before reduction.

### FlagGems-op evidence

Question: what does a FlagGems API op, such as `torch.sum` under
`flag_gems.use_gems()`, accept in tests?

Use the test entry as the boundary. For each tested FlagGems op, inspect:

- pytest marker and test function
- parameter sets such as `REDUCTION_SHAPES`, `FLOAT_DTYPES`, `DIMS_LIST`
- dtype/shape/dim/keepdim at the FlagGems API call
- explicit dtype conversions applied before entering the FlagGems API
- implementation-internal dtype/shape normalization, as notes only

If the test explicitly converts an argument before the FlagGems API call, only
the converted dtype is marked as supported and the row is annotated with
`强制转化为 <dtype>`. A conversion inside the FlagGems implementation does not
replace the API input dtype; it is recorded only as an implementation note.

## Evidence Chain

Triton-op chain:

1. Triton language inventory: establish that the `tl.*` op exists.
2. Direct Triton test mapping: locate tests dedicated to the op.
3. Triton call site: inspect the actual operand passed into `tl.*`.
4. Aggregate dtype and shape evidence at that call boundary.

FlagGems-op chain:

1. FlagGems test marker: find the dedicated tested API op.
2. Test parameters: resolve dtype/shape/dim inputs.
3. API call boundary: resolve explicit conversions applied to call arguments.
4. FlagGems implementation: record internal conversion and shape notes.
5. Stop at the FlagGems implementation; do not inspect Triton language
   implementation details for this table.

## Current Triton Matrix Script Contract

For `scripts/generate_triton_level_support_matrix.py`, the Triton-op matrix
is currently generated from direct Triton tests only:

- scan scope: `python/**/test_*.py`
- operator inventory: fixed Triton language operator list
- output rows: every operator in the inventory is emitted
- evidence boundary: actual `tl.*` call sites inside mapped tests
- output files: `/tmp/triton_level_support_matrix.csv`,
  `/tmp/triton_level_support_evidence.csv`, `/tmp/triton_op_test_mapping.csv`

FlagGems-derived evidence is not mixed into this Triton-op table. It belongs in
the FlagGems-op table, where the boundary is the tested API input.

## Current FlagGems Matrix Script Contract

For `scripts/generate_flaggems_operator_support_matrix.py`, the FlagGems-op
matrix is generated from API-level correctness tests only:

- scan scope: `FlagGems/tests/**/*.py`
- excluded scope: benchmark data, generated caches, result folders, `__pycache__`
- operator inventory: fixed FlagGems/PyTorch API operator list in the script
- evidence boundary: test entry input dtype, shape, dim, layout, and output form
- implementation pass: locate the matching `FlagGems/src/flag_gems/**`
  implementation, then record short notes for internal dtype conversion and
  shape/dim normalization; `tl.*` names may be listed only when they appear in
  that FlagGems source
- stop boundary: do not follow `tl.*` calls into `python/triton/language/**`
- output rows: every operator in the inventory is emitted, including operators
  with no dedicated test
- output files: `/tmp/flaggems_operator_support_matrix.csv`,
  `/tmp/flaggems_operator_support_evidence.csv`,
  `/tmp/flaggems_op_test_mapping_strict.csv`,
  `/tmp/flaggems_operator_implementation_map.csv`

FlagGems-op support is API-call-level support. For
`flag_gems_op(x.to(torch.float32))`, only `fp32` is marked and the row notes
`强制转化为 fp32`. For `flag_gems_op(x)` followed by an internal `fp16 -> fp32`
conversion, the row keeps the tested API input dtype and records the internal
conversion only as a note. Do not use either case as native Triton-op evidence.

## FlagGems Test Mapping Rules

Only count a test for a FlagGems op when the test is dedicated to that op:

- pytest marker exactly matches the op, for example `pytest.mark.gather`
- test function name exactly matches the op after removing `test_` and
  `accuracy_`
- explicit shared-body rules handle cases where one test body validates several
  overload forms, for example `fill_scalar` / `fill_tensor`
- explicit indirect rules are allowed when the test body executes the registered
  op through PyTorch machinery, for example `gather_backward` via
  `torch.autograd.grad` inside the `gather` test

Do not use prefix or substring matching. `zeros` and `zeros_like`, `copy` and
`copy_`, `index` and `index_select`, or `where_self` and `where_self_out` are
separate rows unless an explicit rule says otherwise.

Known current mappings:

- `copy`: no direct API test found; implementation is a functional wrapper over
  `copy_`
- `copy_`: mapped to `pytest.mark.copy_`
- `to_copy`: mapped to `pytest.mark.to_copy`, covering aten `_to_copy`
- `cumsum_out`: implementation and registration exist, but no dedicated
  `cumsum.out` test is currently mapped
- `gather_backward`: mapped indirectly from `test_accuracy_gather`, because that
  test calls `torch.autograd.grad`

## Direct Triton Test Mapping Rules

Only count a test as evidence for an op when the test is dedicated to that op or
the dynamic template explicitly parameterizes that op.

Do not count incidental `tl.*` calls inside another op's test. For example,
`tl.load`, `tl.store`, `tl.arange`, masks, casts, or pointer arithmetic inside a
reduction test are supporting implementation details, not standalone evidence
for those helper ops.

Known exclusions:

- `test_load_reduce`: composite load + reduction + store scenario. It can be
  useful as integration coverage, but it should not be split into standalone
  `load`, `max`, `store`, or `arange` evidence.
- `test_reduce_layouts`: IR/layout/lowering test that builds TTGIR text with
  `tt.reduce`, `tt.load`, `tt.store`, and layout conversions. It is not Triton
  language API evidence for `sum` or `max`.

## Dynamic Template Rules

Tests that patch code with strings such as `tl.{op}` or `tl.atomic_{op}` must be
resolved per parameter row. Do not use the whole test function's dtype/rank
context for every op in the family.

Examples:

- `test_atomic_rmw`: `float16` belongs to `atomic_add` only. `atomic_max` and
  `atomic_min` get only their own parameterized dtypes: integers, `float32`, and
  `float64`.
- `test_reduce`: `reduce_bool` belongs only to `xor_sum`. `sum`, `max`, `min`,
  `argmin`, and `argmax` must not inherit `bool` from that shared test function.
- `test_reduce1d`: the dynamic op list is dedicated to one-dimensional reduction
  configs and can be split across `sum`, `max`, `min`, `argmin`, and `argmax`.
- `test_scan2d`: the dynamic op list is dedicated to scan configs and can be
  split across `associative_scan`, `cumprod`, and `cumsum`.

## Shape And Rank Rules

Shape evidence should come from actual tensor shapes or operands at the `tl.*`
boundary, not from arbitrary numeric parameter tuples.

Rules:

- Prefer concrete tensor constructors such as `torch.randn(shape)`,
  `torch.randint(..., size)`, `numpy_random(shape, ...)`, `zeros`, `empty`, and
  `full`.
- Resolve symbolic dimensions from pytest parametrization when they feed a
  tensor shape. Example: in `test_histogram`, `M` from
  `parametrize("M, N", ...)` feeds `torch.randint(..., (M,), ...)`, so the
  `histogram` input shapes are `(2048,)`, `(1024,)`, `(256,)`, `(32,)`, `(8,)`.
- Do not treat pure parameter rows such as `[2048, 2]` for `("M, N")` as a 2D
  tensor shape. In `test_histogram`, `N` is the number of bins, not a tensor
  dimension for the `tl.histogram` input.
- Rank coverage in the CSV should be derived from concrete `Shape Size` when
  possible. Preserve special markers such as `broadcast` and `non-contiguous`
  only when the tested path actually exercises that layout behavior.
- If a shape is transformed before the `tl.*` call, record the shape/rank at the
  Triton boundary. Keep API-level original shape for the FlagGems-op table.

## Current Output CSV Contract

The Triton-op CSV columns are:

`分类`, `算子`, `uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`,
`uint64`, `int64`, `fp8`, `fp16`, `fp32`, `fp64`, `complex64`, `bf16`,
`bool/int1`, `不支持原因`, `维度覆盖`, `Shape Size`.

`不支持原因` may be empty for now, but the column must remain present.
`Shape Size` should contain concrete shapes when available.

For the FlagGems-op CSV, keep the same column order but use simpler display
rules and append a final `备注` column:

- `不支持原因`: leave blank for mapped/tested rows; write only `无测试` when no
  dedicated or explicit indirect test evidence is mapped
- `维度覆盖`: write compact rank spans such as `0D-5D`; keep `broadcast` and
  `non-contiguous` as extra suffixes only when those paths are directly tested
- `Shape Size`: keep at most one representative concrete shape per rank
- `备注`: record pre-entry forced conversions once, using
  `强制转化为 <dtype>`; keep other repetitive details in the evidence CSV

## Rule Of Thumb

- Triton-op row answers: "what has `tl.*` itself been exercised with?"
- FlagGems-op row answers: "what dtype and shape actually entered the FlagGems
  API in its dedicated tests?"
- Conversion before `tl.*` weakens Triton native support evidence, but not
  FlagGems API support evidence.
- Conversion before the FlagGems API replaces the original dtype evidence and
  must be marked as forced.
