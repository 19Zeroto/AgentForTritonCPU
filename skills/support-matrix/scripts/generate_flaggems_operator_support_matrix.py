#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import itertools
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parent
AGENTFORTRITONCPU_DIR = SKILL_DIR.parent.parent.parent
AGENT_DIR = Path(
    os.environ.get("AGENT_DIR", Path.home() / "agent")
).expanduser().resolve()
ROOT = Path(
    os.environ.get("TRITON_REPO_DIR", AGENT_DIR / "triton-cpu")
).expanduser().resolve()
OUT_DIR = Path(os.environ.get("OUT_DIR", "/tmp")).expanduser().resolve()

OPS = """
cumsum_out
index
layer_norm_backward
mean_dim
normed_cumsum
scaled_softmax_backward
scaled_softmax_forward
softmax_backward
std
trace
erf
erf_
resolve_conj
resolve_neg
arange
arange_start
cat
contiguous
copy
copy_
count_nonzero
diag
diag_embed
fill_scalar
fill_scalar_
fill_tensor
fill_tensor_
flip
full
full_like
gather
gather_backward
hstack
index_add
index_add_
index_put
index_put_
index_select
isin
linspace
logspace
masked_fill
masked_fill_
masked_scatter
masked_scatter_
masked_select
min
nonzero
ones
ones_like
quantile
repeat
repeat_interleave_self_int
repeat_interleave_self_tensor
repeat_interleave_tensor
scatter_add_
select_scatter
slice_scatter
sort
sort_stable
stack
tile
to_copy
topk
vstack
where_self
where_self_out
zeros
zeros_like
""".split()

OUTPUT_DTYPE_COLS = [
    "uint8",
    "int8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "uint64",
    "int64",
    "fp8",
    "fp16",
    "fp32",
    "fp64",
    "complex64",
    "bf16",
    "bool/int1",
]

RANK_COLS = ["scalar", "1D", "2D", "3D", "4D", "5D+", "broadcast", "non-contiguous"]

GROUPS = {
    "Attention/Norm": {
        "layer_norm_backward",
        "normed_cumsum",
        "scaled_softmax_backward",
        "scaled_softmax_forward",
        "softmax_backward",
    },
    "Reduction": {
        "cumsum_out",
        "mean_dim",
        "std",
        "trace",
        "count_nonzero",
        "min",
        "nonzero",
        "quantile",
        "sort",
        "sort_stable",
        "topk",
    },
    "Unary/Elementwise": {
        "erf",
        "erf_",
        "resolve_conj",
        "resolve_neg",
        "contiguous",
        "copy",
        "copy_",
        "fill_scalar",
        "fill_scalar_",
        "fill_tensor",
        "fill_tensor_",
        "flip",
        "masked_fill",
        "masked_fill_",
        "to_copy",
        "where_self",
        "where_self_out",
    },
    "Creation": {
        "arange",
        "arange_start",
        "full",
        "full_like",
        "linspace",
        "logspace",
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
    },
    "Indexing/Shape": {
        "index",
        "cat",
        "diag",
        "diag_embed",
        "gather",
        "gather_backward",
        "hstack",
        "index_add",
        "index_add_",
        "index_put",
        "index_put_",
        "index_select",
        "isin",
        "masked_scatter",
        "masked_scatter_",
        "masked_select",
        "repeat",
        "repeat_interleave_self_int",
        "repeat_interleave_self_tensor",
        "repeat_interleave_tensor",
        "scatter_add_",
        "select_scatter",
        "slice_scatter",
        "stack",
        "tile",
        "vstack",
    },
}

OP_TO_GROUP = {op: group for group, ops in GROUPS.items() for op in ops}

DTYPE_MACROS = {
    "FLOAT_DTYPES": {"fp16", "fp32", "bf16"},
    "ALL_FLOAT_DTYPES": {"fp16", "fp32", "bf16", "fp64"},
    "PRIMARY_FLOAT_DTYPES": {"fp16", "fp32"},
    "INT_DTYPES": {"int16", "int32"},
    "ALL_INT_DTYPES": {"int16", "int32", "int64"},
    "BOOL_TYPES": {"bool/int1"},
    "COMPLEX_DTYPES": {"complex64"},
}

DTYPE_NAMES = {
    "bool": "bool/int1",
    "int1": "bool/int1",
    "uint8": "uint8",
    "int8": "int8",
    "uint16": "uint16",
    "int16": "int16",
    "uint32": "uint32",
    "int32": "int32",
    "uint64": "uint64",
    "int64": "int64",
    "float8e4b15": "fp8",
    "float8e4nv": "fp8",
    "float8e4b8": "fp8",
    "float8e5": "fp8",
    "float8e5b16": "fp8",
    "float16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
    "float32": "fp32",
    "float": "fp32",
    "float64": "fp64",
    "double": "fp64",
    "complex32": "complex64",
    "complex64": "complex64",
    "cfloat": "complex64",
}

TL_TYPE_NAMES = {
    "bool",
    "int1",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float8e4b15",
    "float8e4nv",
    "float8e4b8",
    "float8e5",
    "float8e5b16",
    "float16",
    "float32",
    "float64",
    "bfloat16",
    "dtype",
    "core",
    "language",
    "math",
}

SHAPE_MACROS = {
    "POINTWISE_SHAPES": [
        "scalar",
        "(1,)",
        "(1024, 1024)",
        "(20, 320, 15)",
        "(16, 128, 64, 60)",
        "(16, 7, 57, 32, 29)",
    ],
    "SPECIAL_SHAPES": [
        "(1,)",
        "(1024, 1024)",
        "(20, 320, 15)",
        "(16, 128, 64, 1280)",
        "(16, 7, 57, 32, 29)",
    ],
    "REDUCTION_SHAPES": ["(1, 2)", "(4096, 256)", "(200, 40999, 3)"],
    "REDUCTION_SMALL_SHAPES": ["(1, 2)", "(4096, 256)", "(200, 2560, 3)"],
    "DISTRIBUTION_SHAPES": ["(20, 320, 15)"],
    "TRACE_SHAPES": [
        "(1, 1)",
        "(5, 5)",
        "(10, 20)",
        "(30, 15)",
        "(1, 100)",
        "(100, 1)",
        "(128, 256)",
        "(256, 128)",
        "(0, 10)",
        "(10, 0)",
        "(1500, 1200)",
    ],
    "QUANTILE_SHAPES": ["(1, 2)", "(4096, 256)", "(200, 2560, 3)", "(10, 64, 196)", "(65535, 1)"],
    "THRESHOLD_SHAPE": ["(1, 2)", "(4096, 256)", "(200, 40999, 3)"],
    "NONZERO_SHAPES": ["(1, 2)", "(4096, 256)", "(200, 40999, 3)", "(2637,)"],
    "UT_SHAPES_1D": ["(1,)", "(16,)", "(64,)", "(256,)", "(1024,)", "(33,)", "(81,)", "(273,)", "(1041,)"],
    "UT_SHAPES_2D": [
        "(1, 1)",
        "(1, 16)",
        "(1, 64)",
        "(1, 1000)",
        "(5, 1)",
        "(5, 16)",
        "(5, 64)",
        "(5, 1000)",
        "(1024, 1)",
        "(1024, 16)",
        "(1024, 64)",
        "(1024, 1000)",
    ],
    "STACK_SHAPES": ["(16,)", "(16, 256)", "(20, 320, 15)"],
    "CAT_SHAPES": [
        "(1, 32)",
        "(8, 32)",
        "(16, 128)",
        "(32, 128)",
        "(1024, 1024)",
        "(1, 1024, 256)",
        "(8, 1024, 256)",
        "(16, 1024, 256)",
        "(16, 320, 15)",
        "(32, 320, 15)",
        "(64, 320, 15)",
        "(16, 128, 64, 64)",
        "(24, 128, 64, 64)",
        "(32, 128, 64, 64)",
    ],
    "REPEAT_INTERLEAVE_SHAPES": [
        "(1024, 1024)",
        "(20, 320, 15)",
        "(16, 128, 64, 60)",
        "(16, 7, 57, 32, 29)",
    ],
    "REGULAR_DIM_SHAPE_STRIDES": ["(1, 1024)", "(10000, 128)"],
    "SHAPE_STRIDES": [
        "(1,)",
        "(1024,)",
        "(1000000,)",
        "(1, 1024)",
        "(10000, 128)",
        "(1024, 1)",
        "(128, 10000)",
        "(20, 320, 15)",
        "(200, 40999, 3)",
        "(320, 20, 15)",
        "(3, 40999, 200)",
    ],
    "IRREGULAR_SHAPE_STRIDES": ["(10, 10, 10, 10, 10)"],
    "INDEX_ACC_SHAPE": [
        "(268435456,)",
        "(65536,)",
        "(32, 32)",
        "(8,)",
        "(2, 8)",
        "(512, 512, 512)",
        "(128,)",
        "(2, 128)",
        "(64, 64, 64)",
    ],
    "INDEX_PUT_SHAPE_ACC_FALSE": [
        "(268435456,)",
        "(65536,)",
        "(32, 32)",
        "(8,)",
        "(2, 8)",
        "(32,)",
        "(512, 512, 512)",
        "(128,)",
        "(2, 128)",
        "(512,)",
        "(64, 64, 64)",
        "(2, 8, 64)",
        "(100,)",
        "(16, 16, 4)",
    ],
    "INDEX_PUT_SHAPE_ACC_TRUE": [
        "(268435456,)",
        "(65536,)",
        "(32, 32)",
        "(8,)",
        "(512, 512, 512)",
        "(128,)",
        "(64, 64, 64)",
        "(2, 8)",
    ],
}

NAME_ALIASES = {
    "layer_norm_backward": {"layernorm_backward"},
}

SHARED_BODY_RULES = {
    "fill_scalar": ("FlagGems/tests/test_special_ops.py", "test_fill", "Test fill.Scalar"),
    "fill_tensor": ("FlagGems/tests/test_special_ops.py", "test_fill", "Test fill.Tensor"),
    "fill_scalar_": ("FlagGems/tests/test_binary_pointwise_ops.py", "test_accuracy_fill_", "Test fill_.Scalar"),
    "fill_tensor_": ("FlagGems/tests/test_binary_pointwise_ops.py", "test_accuracy_fill_", "Test fill_.Tensor"),
    "arange_start": ("FlagGems/tests/test_special_ops.py", "test_arange", "torch.arange("),
    "sort_stable": ("FlagGems/tests/test_special_ops.py", "test_sort", "stable=True"),
}

INDIRECT_TEST_RULES = {
    "gather_backward": (
        "FlagGems/tests/test_reduction_ops.py",
        "test_accuracy_gather",
        "autograd backward",
        "torch.autograd.grad",
    ),
}

NO_TEST_NOTES = {
    "copy": "无 copy 专测；实现包装 copy_",
    "cumsum_out": "无 cumsum.out 专测",
}

SOURCE_ALIASES = {
    "mean_dim": {"mean", "mean_dim"},
    "fill_scalar": {"fill"},
    "fill_tensor": {"fill"},
    "fill_scalar_": {"fill_"},
    "fill_tensor_": {"fill_"},
    "sort_stable": {"sort"},
    "where_self_out": {"where_self_out", "where_self"},
    "arange_start": {"arange_start"},
}

FLAGGEMS_ROOTS = [
    ROOT / "FlagGems/src/flag_gems/ops",
    ROOT / "FlagGems/src/flag_gems/fused",
    ROOT / "FlagGems/src/flag_gems/experimental_ops",
    ROOT / "FlagGems/src/flag_gems/runtime/backend/_kunpeng/ops",
    ROOT / "FlagGems/src/flag_gems/utils",
]

EXCLUDE_PARTS = {"benchmark", "results", "code_cache", "__pycache__"}


@dataclass(frozen=True)
class TestContext:
    source: str
    function: str
    line: int
    marks: set[str]
    text: str


@dataclass
class Evidence:
    op: str
    source: str
    function: str
    line: int
    match: str
    dtypes: set[str] = field(default_factory=set)
    ranks: set[str] = field(default_factory=set)
    shapes: set[str] = field(default_factory=set)
    notes: set[str] = field(default_factory=set)


@dataclass
class ImplementationInfo:
    notes: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    triton_ops: set[str] = field(default_factory=set)


@dataclass
class Row:
    group: str
    op: str
    dtypes: set[str] = field(default_factory=set)
    ranks: set[str] = field(default_factory=set)
    shapes: set[str] = field(default_factory=set)
    notes: set[str] = field(default_factory=set)
    tests: set[str] = field(default_factory=set)
    implementation_sources: set[str] = field(default_factory=set)
    triton_ops: set[str] = field(default_factory=set)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def compact(items: set[str] | list[str], limit: int = 100) -> str:
    vals = sorted(str(item) for item in items if str(item))
    if len(vals) <= limit:
        return "; ".join(vals)
    return "; ".join(vals[:limit]) + f"; ... (+{len(vals) - limit})"


def chain(node: ast.AST) -> list[str]:
    out: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        out.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        out.append(cur.id)
        return list(reversed(out))
    return []


def marker_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    out: set[str] = set()
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        parts = chain(target)
        if len(parts) >= 3 and parts[0] == "pytest" and parts[1] == "mark":
            out.add(parts[2])
    return out


def line_context(lines: list[str], node: ast.AST) -> str:
    start = getattr(node, "lineno", 1)
    decs = getattr(node, "decorator_list", [])
    if decs:
        start = min(getattr(dec, "lineno", start) for dec in decs)
    end = getattr(node, "end_lineno", start)
    return "\n".join(lines[start - 1 : end])


def strip_prefixes(name: str) -> str:
    changed = True
    prefixes = ("test_", "accuracy_")
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def exact_name_match(op: str, function: str) -> bool:
    names = {op, *NAME_ALIASES.get(op, set())}
    return strip_prefixes(function) in names


def valid_source(path: Path) -> bool:
    if EXCLUDE_PARTS & set(path.parts):
        return False
    if "runtime" in path.parts and "_kunpeng" not in path.parts:
        return False
    return path.suffix == ".py"


def parse_file(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
    except SyntaxError:
        return None


def scan_tests() -> list[TestContext]:
    out: list[TestContext] = []
    for path in sorted((ROOT / "FlagGems/tests").glob("**/*.py")):
        if EXCLUDE_PARTS & set(path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        tree = parse_file(path)
        if tree is None:
            continue
        lines = text.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            out.append(
                TestContext(
                    source=rel(path),
                    function=node.name,
                    line=node.lineno,
                    marks=marker_names(node),
                    text=line_context(lines, node),
                )
            )
    return out


def is_negative_test(test: TestContext) -> bool:
    lowered_name = test.function.lower()
    if "exception" in lowered_name or "_error" in lowered_name or "error_" in lowered_name:
        return True
    return "pytest.raises" in test.text


def test_dtype_profile(text: str) -> tuple[set[str], set[str]]:
    dtypes = dtype_categories(text)
    forced_dtypes = forced_flaggems_entry_dtypes(text)
    if not forced_dtypes:
        return dtypes, set()
    return forced_dtypes, {f"强制转化为 {compact(forced_dtypes)}"}


def evidence_from_test(op: str, test: TestContext, match: str) -> Evidence:
    dtypes, notes = test_dtype_profile(test.text)
    return Evidence(
        op=op,
        source=test.source,
        function=test.function,
        line=test.line,
        match=match,
        dtypes=dtypes,
        ranks=rank_categories(test.text, op),
        shapes=shape_samples(test.text, op),
        notes=notes,
    )


def collect_mappings(tests: list[TestContext]) -> list[Evidence]:
    evidence: list[Evidence] = []
    by_key = {(test.source, test.function): test for test in tests}
    positive_tests = [test for test in tests if not is_negative_test(test)]
    for op in OPS:
        for test in positive_tests:
            reasons: list[str] = []
            if op in test.marks:
                reasons.append("exact mark")
            if exact_name_match(op, test.function):
                reasons.append("exact test name")
            shared = SHARED_BODY_RULES.get(op)
            if shared and (test.source, test.function) == (shared[0], shared[1]) and shared[2] in test.text:
                reasons.append(shared[2])
            if not reasons:
                continue
            evidence.append(evidence_from_test(op, test, "; ".join(reasons)))
    for op, (source, function, reason) in SHARED_BODY_RULES.items():
        if any(ev.op == op for ev in evidence):
            continue
        test = by_key.get((source, function))
        if not test or is_negative_test(test):
            continue
        evidence.append(evidence_from_test(op, test, reason))
    for op, (source, function, reason, required) in INDIRECT_TEST_RULES.items():
        if any(ev.op == op for ev in evidence):
            continue
        test = by_key.get((source, function))
        if not test or is_negative_test(test) or required not in test.text:
            continue
        evidence.append(evidence_from_test(op, test, reason))
    return evidence


def dtype_expr_categories(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return set(DTYPE_MACROS.get(node.id, set()))
    if isinstance(node, ast.Attribute):
        parts = chain(node)
        if parts:
            dtype = DTYPE_NAMES.get(parts[-1].lower())
            return {dtype} if dtype else set()
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        dtype = DTYPE_NAMES.get(node.value.lower())
        return {dtype} if dtype else set()
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: set[str] = set()
        for elt in node.elts:
            out.update(dtype_expr_categories(elt))
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return dtype_expr_categories(node.left) | dtype_expr_categories(node.right)
    if isinstance(node, ast.Call):
        out: set[str] = set()
        for arg in node.args:
            out.update(dtype_expr_categories(arg))
        return out
    return set()


def parametrize_names_from_call(node: ast.Call) -> list[str]:
    if not node.args:
        return []
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return [name.strip() for name in first.value.split(",")]
    if isinstance(first, (ast.List, ast.Tuple)):
        return [elt.value for elt in first.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)]
    return []


def parametrized_dtype_categories(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        func = ".".join(chain(node.func))
        if not func.endswith("parametrize"):
            continue
        names = parametrize_names_from_call(node)
        if not any("dtype" in name.lower() for name in names):
            continue
        out.update(dtype_expr_categories(node.args[1]))
    return out


def tensor_constructor_dtype_categories(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    default_fp32_constructors = {"rand", "randn", "empty", "empty_strided", "zeros", "ones", "full"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts = chain(node.func)
        if not parts:
            continue
        base = parts[-1]
        kw_dtypes = set()
        for kw in node.keywords:
            if kw.arg == "dtype":
                kw_dtypes.update(dtype_expr_categories(kw.value))
        if kw_dtypes:
            out.update(kw_dtypes)
        elif base in default_fp32_constructors:
            out.add("fp32")
    return out


def dtype_categories(text: str) -> set[str]:
    lowered = text.lower()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    if tree is not None:
        param_dtypes = parametrized_dtype_categories(tree)
        if param_dtypes:
            if "float8" in lowered or "fp8" in lowered:
                param_dtypes.add("fp8")
            return param_dtypes
        out = tensor_constructor_dtype_categories(tree)
    else:
        out = set()
    if not out:
        for macro, dtypes in DTYPE_MACROS.items():
            if macro in text:
                out.update(dtypes)
    if "float8" in lowered or "fp8" in lowered:
        out.add("fp8")
    return out


CAST_METHOD_DTYPES = {
    "bool": "bool/int1",
    "byte": "uint8",
    "short": "int16",
    "int": "int32",
    "long": "int64",
    "half": "fp16",
    "bfloat16": "bf16",
    "float": "fp32",
    "double": "fp64",
    "cfloat": "complex64",
}


def is_use_gems_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and chain(node.func) == ["flag_gems", "use_gems"]


def is_direct_flaggems_api_call(node: ast.Call) -> bool:
    parts = chain(node.func)
    return bool(parts) and parts[0] == "flag_gems" and not is_use_gems_call(node)


def is_dtype_conversion_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    return node.func.attr in {"to", "type", *CAST_METHOD_DTYPES}


def explicit_conversion_dtypes(node: ast.Call) -> set[str]:
    if not isinstance(node.func, ast.Attribute):
        return set()
    method = node.func.attr
    if method in CAST_METHOD_DTYPES:
        return {CAST_METHOD_DTYPES[method]}
    if method not in {"to", "type"}:
        return set()
    out: set[str] = set()
    for arg in node.args:
        out.update(dtype_expr_categories(arg))
    for kw in node.keywords:
        if kw.arg == "dtype":
            out.update(dtype_expr_categories(kw.value))
    return out


def forced_dtypes_from_expr(node: ast.AST, env: dict[str, set[str]]) -> set[str]:
    if isinstance(node, ast.Name):
        return set(env.get(node.id, set()))
    if isinstance(node, ast.Call):
        out = explicit_conversion_dtypes(node)
        if isinstance(node.func, ast.Attribute):
            out.update(forced_dtypes_from_expr(node.func.value, env))
        for arg in node.args:
            out.update(forced_dtypes_from_expr(arg, env))
        for kw in node.keywords:
            out.update(forced_dtypes_from_expr(kw.value, env))
        return out
    if isinstance(node, ast.Attribute):
        return forced_dtypes_from_expr(node.value, env)
    if isinstance(node, ast.Subscript):
        return forced_dtypes_from_expr(node.value, env)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: set[str] = set()
        for elt in node.elts:
            out.update(forced_dtypes_from_expr(elt, env))
        return out
    return set()


def assign_forced_dtypes(target: ast.AST, dtypes: set[str], env: dict[str, set[str]]) -> None:
    if isinstance(target, ast.Name):
        if dtypes:
            env[target.id] = set(dtypes)
        else:
            env.pop(target.id, None)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            assign_forced_dtypes(elt, set(), env)


def is_non_api_helper_call(node: ast.Call) -> bool:
    parts = chain(node.func)
    if not parts:
        return False
    if parts[0] in {"pytest", "math", "np"}:
        return True
    if parts[:2] == ["torch", "testing"]:
        return True
    if parts[0].startswith("gems_assert"):
        return True
    return parts[0] in {"len", "range", "enumerate", "zip", "list", "tuple", "set", "dict"}


def forced_dtypes_in_gems_call(node: ast.Call, env: dict[str, set[str]]) -> set[str]:
    if is_dtype_conversion_call(node) or is_use_gems_call(node) or is_non_api_helper_call(node):
        return set()
    out: set[str] = set()
    if isinstance(node.func, ast.Attribute):
        out.update(forced_dtypes_from_expr(node.func.value, env))
    for arg in node.args:
        out.update(forced_dtypes_from_expr(arg, env))
    for kw in node.keywords:
        out.update(forced_dtypes_from_expr(kw.value, env))
    return out


def forced_flaggems_entry_dtypes(text: str) -> set[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    env: dict[str, set[str]] = {}
    forced: set[str] = set()

    def visit_statements(statements: list[ast.stmt], in_gems: bool = False) -> None:
        for stmt in statements:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call) and (in_gems or is_direct_flaggems_api_call(node)):
                    forced.update(forced_dtypes_in_gems_call(node, env))

            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                value = stmt.value
                value_dtypes = forced_dtypes_from_expr(value, env) if value is not None else set()
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for target in targets:
                    assign_forced_dtypes(target, value_dtypes, env)
            elif isinstance(stmt, ast.AugAssign):
                assign_forced_dtypes(stmt.target, set(), env)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_statements(stmt.body, in_gems)
            elif isinstance(stmt, ast.With):
                nested_in_gems = in_gems or any(is_use_gems_call(item.context_expr) for item in stmt.items)
                visit_statements(stmt.body, nested_in_gems)
            elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.If, ast.Try)):
                bodies = [getattr(stmt, "body", []), getattr(stmt, "orelse", [])]
                if isinstance(stmt, ast.Try):
                    bodies.extend(handler.body for handler in stmt.handlers)
                    bodies.append(stmt.finalbody)
                for body in bodies:
                    visit_statements(body, in_gems)

    visit_statements(tree.body)
    return forced


def rank_from_len(n: int) -> str:
    if n <= 0:
        return "scalar"
    if n == 1:
        return "1D"
    if n == 2:
        return "2D"
    if n == 3:
        return "3D"
    if n == 4:
        return "4D"
    return "5D+"


def shape_rank(shape: str) -> str | None:
    if shape in {"scalar", "()"}:
        return "scalar"
    nums = re.findall(r"-?\d+", shape)
    return rank_from_len(len(nums)) if nums else None


def shape_from_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Tuple) and not node.elts:
        return "scalar"
    if isinstance(node, (ast.Tuple, ast.List)):
        values: list[int] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, int) and not isinstance(elt.value, bool):
                values.append(elt.value)
            elif isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.USub) and isinstance(elt.operand, ast.Constant) and isinstance(elt.operand.value, int):
                values.append(-elt.operand.value)
            else:
                return None
        if not values:
            return "scalar"
        if len(values) == 1:
            return f"({values[0]},)"
        return "(" + ", ".join(str(v) for v in values) + ")"
    return None


def format_shape(values: list[int]) -> str:
    if not values:
        return "scalar"
    if len(values) == 1:
        return f"({values[0]},)"
    return "(" + ", ".join(str(v) for v in values) + ")"


def literal_shapes(node: ast.AST) -> set[str]:
    out: set[str] = set()
    shape = shape_from_literal(node)
    if shape:
        out.add(shape)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for elt in node.elts:
            out.update(literal_shapes(elt))
    elif isinstance(node, ast.BinOp):
        out.update(literal_shapes(node.left))
        out.update(literal_shapes(node.right))
    elif isinstance(node, ast.IfExp):
        out.update(literal_shapes(node.body))
        out.update(literal_shapes(node.orelse))
    elif isinstance(node, ast.ListComp):
        out.update(literal_shapes(node.elt))
        for gen in node.generators:
            out.update(literal_shapes(gen.iter))
    return out


def numeric_expr_values(node: ast.AST, params: dict[str, list[int | float | None]] | None = None) -> list[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return [node.value]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return [-value for value in numeric_expr_values(node.operand, params)]
    if isinstance(node, ast.Name) and params:
        return [int(value) for value in params.get(node.id, []) if isinstance(value, int)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub, ast.FloorDiv)):
        left = numeric_expr_values(node.left, params)
        right = numeric_expr_values(node.right, params)
        out: list[int] = []
        for lhs in left:
            for rhs in right:
                if isinstance(node.op, ast.Mult):
                    out.append(lhs * rhs)
                elif isinstance(node.op, ast.Add):
                    out.append(lhs + rhs)
                elif isinstance(node.op, ast.Sub):
                    out.append(lhs - rhs)
                elif isinstance(node.op, ast.FloorDiv) and rhs:
                    out.append(lhs // rhs)
        return out
    return []


def symbolic_shapes(node: ast.AST, params: dict[str, list[int | float | None]]) -> set[str]:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return set()
    dim_values = [numeric_expr_values(elt, params) for elt in node.elts]
    if not dim_values or any(not values for values in dim_values):
        return set()
    return {format_shape([int(value) for value in values]) for values in itertools.product(*dim_values)}


def constant_values(node: ast.AST) -> list[int | float | None]:
    if isinstance(node, ast.Constant):
        return [node.value] if isinstance(node.value, (int, float)) or node.value is None else []
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub)):
        left = constant_values(node.left)
        right = constant_values(node.right)
        out: list[int | float | None] = []
        for lhs in left:
            for rhs in right:
                if not isinstance(lhs, (int, float)) or not isinstance(rhs, (int, float)):
                    continue
                if isinstance(node.op, ast.Mult):
                    out.append(lhs * rhs)
                elif isinstance(node.op, ast.Add):
                    out.append(lhs + rhs)
                elif isinstance(node.op, ast.Sub):
                    out.append(lhs - rhs)
        return out
    if isinstance(node, ast.IfExp):
        return constant_values(node.body) + constant_values(node.orelse)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: list[int | float | None] = []
        for elt in node.elts:
            out.extend(constant_values(elt))
        return out
    return []


def parametrize_map(text: str) -> dict[str, list[int | float | None]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    params: dict[str, list[int | float | None]] = defaultdict(list)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        func = ".".join(chain(node.func))
        if not func.endswith("parametrize"):
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        names = [name.strip() for name in first.value.split(",")]
        if len(names) == 1:
            params[names[0]].extend(constant_values(node.args[1]))
    return params


def shape_args_from_call(node: ast.Call) -> list[ast.AST]:
    parts = chain(node.func)
    if not parts:
        return []
    base = parts[-1]
    args: list[ast.AST] = []
    if base == "randint":
        if len(node.args) >= 3:
            args.append(node.args[2])
    elif base in {"rand", "randn", "zeros", "ones", "empty", "full", "empty_strided", "numpy_random", "broadcast_to"}:
        if node.args:
            args.append(node.args[0])
    for kw in node.keywords:
        if kw.arg in {"shape", "size"}:
            args.append(kw.value)
    return args


def shape_samples(text: str, op: str) -> set[str]:
    out: set[str] = set()
    for macro, shapes in SHAPE_MACROS.items():
        if macro in text:
            out.update(shapes)
    if "get_diag_embed_shape_and_dims" in text:
        out.update({"(1024,)", "(1024, 1024)"})
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    params = parametrize_map(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = ".".join(chain(node.func))
            if len(node.args) >= 2 and func.endswith("parametrize") and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                names = parametrize_names_from_call(node)
                if any("shape" in name or name in {"size", "sizes"} for name in names):
                    out.update(literal_shapes(node.args[1]))
            for shape_arg in shape_args_from_call(node):
                out.update(literal_shapes(shape_arg))
                out.update(symbolic_shapes(shape_arg, params))
        elif isinstance(node, ast.Assign):
            target_names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if any("shape" in name for name in target_names):
                out.update(literal_shapes(node.value))
    if op in {"linspace", "logspace"}:
        for steps in params.get("steps", []):
            if isinstance(steps, int):
                out.add(f"({steps},)")
    if op in {"arange", "arange_start"}:
        starts = [v for v in params.get("start", [0]) if isinstance(v, (int, float))]
        ends = [v for v in params.get("end", []) if isinstance(v, (int, float))]
        steps = [v for v in params.get("step", [1]) if isinstance(v, (int, float)) and v != 0]
        if not starts:
            starts = [0]
        for start in starts:
            for end in ends:
                for step in steps:
                    size = math.ceil((end - start) / step)
                    if size > 0:
                        out.add(f"({int(size)},)")
    if "batch_size" in params and "hiddensize" in params:
        for batch in params["batch_size"]:
            for hidden in params["hiddensize"]:
                if isinstance(batch, int) and isinstance(hidden, int):
                    out.add(f"({batch}, {hidden})")
    if "broadcast" in text.lower():
        out.add("broadcast")
    if re.search(r"non[_ -]?contiguous|\[::|as_strided|permute|transpose|\.t\s*\(", text, re.I):
        out.add("non-contiguous")
    return out


def rank_categories(text: str, op: str) -> set[str]:
    ranks: set[str] = set()
    for shape in shape_samples(text, op):
        rank = shape_rank(shape)
        if rank:
            ranks.add(rank)
        if shape == "broadcast":
            ranks.add("broadcast")
        if shape == "non-contiguous":
            ranks.add("non-contiguous")
    if "broadcast" in text.lower():
        ranks.add("broadcast")
    if re.search(r"non[_ -]?contiguous|\[::|as_strided|permute|transpose|\.t\s*\(", text, re.I):
        ranks.add("non-contiguous")
    return ranks


def public_defs(path: Path) -> set[str]:
    tree = parse_file(path)
    names = {path.stem}
    if tree is None:
        return names
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def flaggems_source_files() -> list[Path]:
    paths: list[Path] = []
    for root in FLAGGEMS_ROOTS:
        if root.exists():
            paths.extend(path for path in root.rglob("*.py") if valid_source(path))
    return sorted(set(paths))


def source_index() -> dict[str, set[Path]]:
    by_name: dict[str, set[Path]] = defaultdict(set)
    for path in flaggems_source_files():
        for name in public_defs(path):
            by_name[name].add(path)
    return by_name


def implementation_info(op: str, by_name: dict[str, set[Path]]) -> ImplementationInfo:
    names = {op, *SOURCE_ALIASES.get(op, set())}
    paths: set[Path] = set()
    for name in names:
        paths.update(by_name.get(name, set()))
    info = ImplementationInfo()
    info.sources.update(rel(path) for path in paths)
    if not paths:
        info.notes.add("未定位实现")
        return info
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in paths)
    info.triton_ops.update(
        name
        for name in re.findall(r"\btl\.([A-Za-z_]\w*)", text)
        if name not in TL_TYPE_NAMES
    )
    if any(
        token in text
        for token in [
            ".to(",
            "astype",
            "compute_dtype",
            "dtype is torch.bool",
            "dtype == torch.bool",
            "torch.float32",
            "torch.int64",
            "tl.float32",
            "upcast",
        ]
    ):
        info.notes.add("实现含 dtype 转换/特殊分支")
    if re.search(r"dim_compress|reshape|view\(|flatten|squeeze|unsqueeze|contiguous|dim\s*=", text):
        info.notes.add("实现含 shape/dim 规整")
    return info


def implementation_index() -> dict[str, ImplementationInfo]:
    by_name = source_index()
    return {op: implementation_info(op, by_name) for op in OPS}


def build_rows(evidence: list[Evidence], impl_by_op: dict[str, ImplementationInfo]) -> list[Row]:
    rows = {op: Row(OP_TO_GROUP.get(op, "Other"), op) for op in OPS}
    for op, row in rows.items():
        info = impl_by_op[op]
        row.notes.update(info.notes)
        row.implementation_sources.update(info.sources)
        row.triton_ops.update(info.triton_ops)
    for ev in evidence:
        row = rows[ev.op]
        row.dtypes.update(ev.dtypes)
        row.ranks.update(ev.ranks)
        row.shapes.update(shape for shape in ev.shapes if shape not in {"broadcast", "non-contiguous"})
        row.notes.update(ev.notes)
        row.tests.add(f"{ev.source}:{ev.line}::{ev.function}")
    for row in rows.values():
        if not row.tests:
            row.notes.add(NO_TEST_NOTES.get(row.op, "未找到专门测试"))
    return [rows[op] for op in OPS]


def compact_ranks(ranks: set[str]) -> str:
    rank_values = {
        "scalar": 0,
        "1D": 1,
        "2D": 2,
        "3D": 3,
        "4D": 4,
        "5D+": 5,
    }
    values = sorted(rank_values[rank] for rank in ranks if rank in rank_values)
    spans: list[str] = []
    start: int | None = None
    prev: int | None = None
    for value in values:
        if start is None:
            start = prev = value
            continue
        if prev is not None and value == prev + 1:
            prev = value
            continue
        spans.append(format_rank_span(start, prev if prev is not None else start))
        start = prev = value
    if start is not None:
        spans.append(format_rank_span(start, prev if prev is not None else start))
    if "broadcast" in ranks:
        spans.append("broadcast")
    if "non-contiguous" in ranks:
        spans.append("non-contiguous")
    return ";".join(spans)


def format_rank_span(start: int, end: int) -> str:
    if start == end:
        return f"{start}D"
    return f"{start}D-{end}D"


def compact_shape_representatives(shapes: set[str]) -> str:
    by_rank: dict[str, list[str]] = defaultdict(list)
    for shape in sorted(shapes):
        rank = shape_rank(shape)
        if rank:
            by_rank[rank].append(shape)
    reps: list[str] = []
    for rank in ["scalar", "1D", "2D", "3D", "4D", "5D+"]:
        vals = by_rank.get(rank)
        if vals:
            reps.append(vals[0])
    return "; ".join(reps)


def unsupported_reason(row: Row) -> str:
    return "" if row.tests else "无测试"


def matrix_notes(row: Row) -> str:
    return compact({note for note in row.notes if note.startswith("强制转化")}, 4)


def write_outputs(rows: list[Row], evidence: list[Evidence], impl_by_op: dict[str, ImplementationInfo]) -> None:
    headers = ["分类", "算子", *OUTPUT_DTYPE_COLS, "不支持原因", "维度覆盖", "Shape Size", "备注"]
    with (OUT_DIR / "flaggems_operator_support_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(
                [
                    row.group,
                    row.op,
                    *["Y" if dtype in row.dtypes else "-" for dtype in OUTPUT_DTYPE_COLS],
                    unsupported_reason(row),
                    compact_ranks(row.ranks),
                    compact_shape_representatives(row.shapes),
                    matrix_notes(row),
                ]
            )
    with (OUT_DIR / "flaggems_operator_support_evidence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "op",
                "source",
                "line",
                "function",
                "match",
                "dtypes",
                "ranks",
                "shapes",
                "notes",
                "implementation_sources",
                "triton_ops",
                "implementation_notes",
            ]
        )
        for ev in evidence:
            info = impl_by_op[ev.op]
            writer.writerow(
                [
                    ev.op,
                    ev.source,
                    ev.line,
                    ev.function,
                    ev.match,
                    compact(ev.dtypes),
                    compact(ev.ranks),
                    compact(ev.shapes),
                    compact(ev.notes),
                    compact(info.sources),
                    compact(info.triton_ops),
                    compact(info.notes, 6),
                ]
            )
    with (OUT_DIR / "flaggems_op_test_mapping_strict.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["op", "source", "line", "function", "match"])
        for ev in evidence:
            writer.writerow([ev.op, ev.source, ev.line, ev.function, ev.match])
    with (OUT_DIR / "flaggems_operator_implementation_map.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["op", "implementation_sources", "triton_ops", "implementation_notes"])
        for row in rows:
            writer.writerow(
                [
                    row.op,
                    compact(row.implementation_sources),
                    compact(row.triton_ops),
                    compact(row.notes, 6),
                ]
            )


def main() -> None:
    tests = scan_tests()
    evidence = collect_mappings(tests)
    impl_by_op = implementation_index()
    rows = build_rows(evidence, impl_by_op)
    write_outputs(rows, evidence, impl_by_op)
    covered = sum(1 for row in rows if row.tests)
    print(f"rows={len(rows)}")
    print(f"covered_rows={covered}")
    print(f"evidence_records={len(evidence)}")
    print(OUT_DIR / "flaggems_operator_support_matrix.csv")
    print(OUT_DIR / "flaggems_operator_support_evidence.csv")
    print(OUT_DIR / "flaggems_op_test_mapping_strict.csv")
    print(OUT_DIR / "flaggems_operator_implementation_map.csv")


if __name__ == "__main__":
    main()
