#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parent
AGENTFORTRITONCPU_DIR = SKILL_DIR.parent.parent.parent
AGENT_DIR = Path(
    os.environ.get("AGENT_DIR", AGENTFORTRITONCPU_DIR.parent)
).expanduser().resolve()
ROOT = Path(
    os.environ.get("TRITON_REPO_DIR", AGENT_DIR / "triton-cpu")
).expanduser().resolve()
OUT_DIR = Path(os.environ.get("OUT_DIR", "/tmp")).expanduser().resolve()

DTYPE_COLS = [
    "bool",
    "int16",
    "int32",
    "int64",
    "uint8",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex",
    "fp8/other",
]

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

RANK_COLS = [
    "scalar",
    "1D",
    "2D",
    "3D",
    "4D",
    "5D+",
    "broadcast",
    "non-contiguous",
]

REFERENCE_GROUPS = [
    ("Programming Model", ["tensor", "tensor_descriptor", "program_id", "num_programs", "map_elementwise"]),
    ("Creation Ops", ["arange", "cat", "full", "zeros", "zeros_like", "cast", "to_tensor"]),
    (
        "Shape Manipulation Ops",
        [
            "broadcast",
            "broadcast_to",
            "expand_dims",
            "interleave",
            "join",
            "permute",
            "ravel",
            "reshape",
            "split",
            "squeeze",
            "trans",
            "unsqueeze",
            "view",
        ],
    ),
    ("Linear Algebra Ops", ["dot", "dot_scaled"]),
    (
        "Memory/Pointer Ops",
        [
            "load",
            "store",
            "make_tensor_descriptor",
            "load_tensor_descriptor",
            "store_tensor_descriptor",
            "make_block_ptr",
            "advance",
        ],
    ),
    ("Indexing Ops", ["flip", "where", "swizzle2d"]),
    (
        "Math Ops",
        [
            "abs",
            "add",
            "cdiv",
            "ceil",
            "clamp",
            "cos",
            "div_rn",
            "erf",
            "exp",
            "exp2",
            "fdiv",
            "floor",
            "fma",
            "log",
            "log2",
            "maximum",
            "minimum",
            "mul",
            "rsqrt",
            "sigmoid",
            "sin",
            "softmax",
            "sqrt",
            "sqrt_rn",
            "sub",
            "umulhi",
        ],
    ),
    (
        "Reduction Ops",
        [
            "argmax",
            "argmin",
            "max",
            "min",
            "reduce",
            "reduce_or",
            "sum",
            "xor_sum",
            "associative_scan",
            "cumprod",
            "cumsum",
            "histogram",
            "sort",
            "topk",
            "gather",
        ],
    ),
    (
        "Atomic Ops",
        [
            "atomic_add",
            "atomic_and",
            "atomic_cas",
            "atomic_max",
            "atomic_min",
            "atomic_or",
            "atomic_poll",
            "atomic_xchg",
            "atomic_xor",
        ],
    ),
    ("Random Number Generation", ["randint4x", "randint", "rand", "rand4x", "randn", "randn4x"]),
    ("Iterators", ["range", "static_range"]),
    ("Inline Assembly", ["inline_asm_elementwise"]),
    ("Compiler Hint Ops", ["assume", "debug_barrier", "max_constancy", "max_contiguous", "multiple_of"]),
    ("Debug Ops", ["static_print", "static_assert", "device_print", "device_assert", "expect_zero"]),
]

OP_TO_GROUP = {op: group for group, ops in REFERENCE_GROUPS for op in ops}
CANONICAL_OPS = set(OP_TO_GROUP)
GROUP_ORDER = {group: i for i, (group, _ops) in enumerate(REFERENCE_GROUPS)}
OP_ORDER = {op: i for _group, ops in REFERENCE_GROUPS for i, op in enumerate(ops)}

DTYPE_CONSTS = {
    "bool": "bool",
    "int1": "bool",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
    "float64": "float64",
    "float8e4b15": "fp8/other",
    "float8e4nv": "fp8/other",
    "float8e4b8": "fp8/other",
    "float8e5": "fp8/other",
    "float8e5b16": "fp8/other",
}

OUTPUT_DTYPE_ALIASES = {
    "uint8": {"uint8"},
    "int8": {"int8"},
    "uint16": {"uint16"},
    "int16": {"int16"},
    "uint32": {"uint32"},
    "int32": {"int32"},
    "uint64": {"uint64"},
    "int64": {"int64"},
    "fp8": {"fp8/other"},
    "fp16": {"float16"},
    "fp32": {"float32"},
    "fp64": {"float64"},
    "complex64": {"complex"},
    "bf16": {"bfloat16"},
    "bool/int1": {"bool"},
}

SYNTAX_OPS = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "fdiv",
}

CALL_ALIASES = {
    "_experimental_descriptor_load": "load_tensor_descriptor",
    "_experimental_descriptor_store": "store_tensor_descriptor",
}

METHOD_ALIASES = {
    "to": "cast",
    "cast": "cast",
    "broadcast_to": "broadcast_to",
    "reshape": "reshape",
    "view": "view",
    "ravel": "ravel",
    "permute": "permute",
    "trans": "trans",
    "split": "split",
    "squeeze": "squeeze",
    "unsqueeze": "unsqueeze",
}

REDUCTION_OPS = {
    "sum",
    "max",
    "min",
    "argmax",
    "argmin",
    "reduce",
    "xor_sum",
    "associative_scan",
    "cumprod",
    "cumsum",
}

ELEMENTWISE_ARG_OPS = {
    "abs",
    "ceil",
    "clamp",
    "cos",
    "div_rn",
    "erf",
    "exp",
    "exp2",
    "fdiv",
    "floor",
    "fma",
    "log",
    "log2",
    "maximum",
    "minimum",
    "mul",
    "rsqrt",
    "sigmoid",
    "sin",
    "softmax",
    "sqrt",
    "sqrt_rn",
    "sub",
    "umulhi",
    "add",
    "where",
}

FLAGGEMS_ROOTS = [
    ROOT / "FlagGems/src/flag_gems/ops",
    ROOT / "FlagGems/src/flag_gems/fused",
    ROOT / "FlagGems/src/flag_gems/experimental_ops",
    ROOT / "FlagGems/src/flag_gems/runtime/backend/_kunpeng/ops",
    ROOT / "FlagGems/src/flag_gems/utils",
]

DIRECT_TEST_ROOTS = [
    ROOT / "python",
]

EXCLUDE_PARTS = {"benchmark", "results", "code_cache", "__pycache__"}

API_TO_TRITON_OPS = {
    "amax": {"max"},
    "amin": {"min"},
    "argmax": {"argmax"},
    "argmin": {"argmin"},
    "add": {"add"},
    "sub": {"sub"},
    "mul": {"mul"},
    "div": {"fdiv", "div_rn"},
    "divide": {"fdiv", "div_rn"},
    "floor_divide": {"fdiv"},
    "maximum": {"maximum"},
    "minimum": {"minimum"},
    "sum": {"sum"},
    "cumsum": {"cumsum"},
    "cumprod": {"cumprod"},
    "sort": {"sort"},
    "topk": {"topk"},
    "gather": {"gather"},
    "where": {"where"},
    "exp": {"exp"},
    "exp2": {"exp2"},
    "log": {"log"},
    "log2": {"log2"},
    "sin": {"sin"},
    "cos": {"cos"},
    "sqrt": {"sqrt"},
    "rsqrt": {"rsqrt"},
    "erf": {"erf"},
    "abs": {"abs"},
    "clamp": {"clamp"},
    "sigmoid": {"sigmoid"},
    "softmax": {"softmax"},
    "rand": {"rand"},
    "randn": {"randn"},
    "randint": {"randint"},
}

DIRECT_TEST_OP_OVERRIDES = {
    "test_math_op": {"ceil", "cos", "exp", "exp2", "floor", "log", "log2", "sin", "sqrt"},
    "test_math_divide_op": {"div_rn", "fdiv"},
    "test_precise_math": {"div_rn", "sqrt_rn"},
    "test_unary_math": {"cos", "exp", "exp2", "log", "log2", "rsqrt", "sin", "sqrt"},
    "test_atomic_rmw": {"atomic_add", "atomic_max", "atomic_min"},
    "test_atomic_rmw_predicate": {"atomic_max"},
    "test_tensor_atomic_rmw": {"atomic_add"},
    "test_tensor_atomic_rmw_block": {"atomic_min"},
    "test_atomic_cas": {"atomic_cas"},
    "test_tensor_atomic_cas": {"atomic_cas"},
    "test_reduce1d": {"argmax", "argmin", "max", "min", "sum"},
    "test_reduce": {"argmax", "argmin", "max", "min", "sum", "xor_sum"},
    "test_scan2d": {"associative_scan", "cumprod", "cumsum"},
}

DIRECT_TEST_EXCLUDES = {
    # Composite/layout tests: useful for integration coverage, but not standalone
    # Triton language op support evidence.
    "test_load_reduce",
    "test_reduce_layouts",
}


@dataclass
class Attrs:
    dtypes: set[str] = field(default_factory=set)
    ranks: set[str] = field(default_factory=set)
    shapes: set[str] = field(default_factory=set)
    notes: set[str] = field(default_factory=set)

    def copy(self) -> "Attrs":
        return Attrs(set(self.dtypes), set(self.ranks), set(self.shapes), set(self.notes))

    def merge(self, other: "Attrs") -> "Attrs":
        self.dtypes.update(other.dtypes)
        self.ranks.update(other.ranks)
        self.shapes.update(other.shapes)
        self.notes.update(other.notes)
        return self


@dataclass
class Evidence:
    op: str
    source_kind: str
    source: str
    function: str
    dtypes: set[str] = field(default_factory=set)
    ranks: set[str] = field(default_factory=set)
    shapes: set[str] = field(default_factory=set)
    api_ops: set[str] = field(default_factory=set)
    notes: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class TestMapping:
    op: str
    source_kind: str
    source: str
    function: str
    api: str = ""
    reason: str = ""


@dataclass
class Row:
    group: str
    op: str
    dtypes: set[str] = field(default_factory=set)
    ranks: set[str] = field(default_factory=set)
    shapes: set[str] = field(default_factory=set)
    direct: set[str] = field(default_factory=set)
    flaggems: set[str] = field(default_factory=set)
    notes: set[str] = field(default_factory=set)
    evidence_count: int = 0


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def compact(items: set[str] | list[str], limit: int = 12) -> str:
    ordered = sorted(str(x) for x in items if str(x))
    if len(ordered) <= limit:
        return "; ".join(ordered)
    return "; ".join(ordered[:limit]) + f"; ... (+{len(ordered) - limit})"


def compact_notes(notes: set[str] | list[str], limit: int = 4) -> str:
    flags: list[str] = []
    raw = sorted(str(note) for note in notes if str(note))

    def add(flag: str) -> None:
        if flag not in flags:
            flags.append(flag)

    for note in raw:
        lowered = note.lower()
        if "promote" in lowered or "promoted" in lowered or "accumulator dtype" in lowered:
            add("dtype promotion/cast")
        elif "converted via" in lowered:
            add("dtype promotion/cast")
        elif "axis/keep_dims" in lowered:
            add("axis variants")
        elif "load dtype inferred" in lowered:
            add("operand dtype inferred")
        elif "store evidence" in lowered:
            add("store value evidence")
        elif "text/template evidence" in lowered:
            add("template-only evidence")
        elif "dtype constant" in lowered:
            add("operand dtype inferred")
        elif "tensor/operator syntax" in lowered:
            add("syntax evidence")
        elif "tensor method" in lowered:
            add("method evidence")
        elif "shape permutation" in lowered:
            add("shape permutation")
        elif "input element dtype" in lowered:
            add("operand dtype inferred")
        elif "scaled dot operand" in lowered:
            add("scaled-dot operand evidence")
        else:
            add(note)

    if len(flags) <= limit:
        return "; ".join(flags)
    return "; ".join(flags[:limit]) + f"; ... (+{len(flags) - limit})"


def dedicated_triton_ops_for_api(api: str) -> set[str]:
    ops = set(API_TO_TRITON_OPS.get(api, set()))
    if api in CANONICAL_OPS:
        ops.add(api)
    return ops


def normalized_words(text: str) -> tuple[str, set[str]]:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return normalized, set(normalized.split())


def dedicated_names_from_parts(parts: list[str], candidates: set[str]) -> set[str]:
    haystack = " ".join(parts)
    normalized, tokens = normalized_words(haystack)
    compact_tokens = {token.replace("_", "") for token in tokens}
    matched: set[str] = set()

    if "dot_scaled" in candidates and (
        "scaled dot" in normalized or "dot scaled" in normalized
    ):
        matched.add("dot_scaled")

    for name in sorted(candidates, key=lambda item: (-item.count("_"), -len(item), item)):
        lowered = name.lower()
        words = lowered.split("_")
        phrase = " ".join(words)
        compact = "".join(words)
        if len(words) > 1:
            if phrase in normalized or compact in compact_tokens:
                matched.add(name)
        elif lowered in tokens:
            matched.add(name)

    longer_words = {
        word
        for name in matched
        if "_" in name
        for word in name.lower().split("_")
    }
    return {
        name
        for name in matched
        if "_" in name or name.lower() not in longer_words
    }


def dedicated_ops_from_direct_context(name: str, source_text: str, path: Path) -> set[str]:
    ops = dedicated_names_from_parts(
        [name, path.stem.removeprefix("test_")],
        CANONICAL_OPS,
    )
    exact_test_name = name.lower().removeprefix("test_")
    exact_file_name = path.stem.lower().removeprefix("test_")
    if "tensor" in ops and "tensor" not in {exact_test_name, exact_file_name}:
        ops.remove("tensor")
    return ops


def summarize_direct_evidence(items: set[str]) -> str:
    if not items:
        return ""
    files = {item.split("::", 1)[0].rsplit("/", 1)[-1] for item in items}
    return f"{len(items)} call sites / {len(files)} files"


def summarize_flaggems_ops(items: set[str], limit: int = 10) -> str:
    if not items:
        return ""
    names = set()
    for item in items:
        name = item.split("(", 1)[0].strip()
        if name:
            names.add(name)
    ordered = sorted(names)
    if len(ordered) <= limit:
        return "; ".join(ordered)
    return "; ".join(ordered[:limit]) + f"; ... (+{len(ordered) - limit})"


def attr_chain(node: ast.AST) -> list[str]:
    chain: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        chain.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        chain.append(cur.id)
        return list(reversed(chain))
    return []


def parse_file(path: Path) -> ast.Module | None:
    try:
        return ast.parse(read_text(path), filename=str(path))
    except SyntaxError:
        return None


def import_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    tl_aliases = {"tl"}
    triton_aliases = {"triton"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if alias.name == "triton.language":
                    tl_aliases.add(name)
                elif alias.name == "triton":
                    triton_aliases.add(name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = alias.asname or alias.name
                if module == "triton" and alias.name == "language":
                    tl_aliases.add(name)
                elif module == "triton.language":
                    tl_aliases.add(name)
                elif module == "triton" and alias.name == "jit":
                    triton_aliases.add(name)
    return tl_aliases, triton_aliases


def decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return decorator_name(node.func)
    chain = attr_chain(node)
    if chain:
        return ".".join(chain)
    if isinstance(node, ast.Name):
        return node.id
    return ""


def decorator_is_jit(decorator: ast.AST, triton_aliases: set[str]) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    chain = attr_chain(target)
    if not chain:
        return False
    return chain[-1] == "jit" and (chain[0] in triton_aliases or len(chain) == 1)


def rank_from_len(n: int) -> str:
    if n == 0:
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
    "DISTRIBUTION_SHAPES": ["(20, 320, 15)"],
    "REDUCTION_SHAPES": ["(1, 2)", "(4096, 256)", "(200, 40999, 3)"],
    "REDUCTION_SMALL_SHAPES": ["(1, 2)", "(4096, 256)", "(200, 2560, 3)"],
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
    "reduce_configs1": ["(1, 1024)"],
    "reduce_configs2": ["(2, 32)", "(4, 32)", "(4, 128)", "(16, 32)"],
    "reduce_configs3": ["(2, 32, 16)", "(32, 2, 16)", "(32, 16, 2)"],
    "invalid_config": ["(32, 32)"],
    "negative_config": ["(32, 32)"],
    "keep_dims_2d_configs": ["(32, 32)"],
    "keep_dims_3d_configs": ["(32, 2, 16)"],
    "reduce_bool": ["(2, 32)", "(4, 32)", "(4, 128)"],
    "scan_configs": ["(1, 1024)", "(2, 1024)", "(8, 32)", "(16, 32)", "(32, 16)", "(32, 32)", "(1024, 2)"],
}

DTYPES_WITH_BFLOAT16 = {
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "bfloat16",
}


def shape_rank(shape: str) -> str | None:
    if shape == "scalar" or shape == "()":
        return "scalar"
    nums = re.findall(r"-?\d+", shape)
    if nums:
        return rank_from_len(len(nums))
    return None


def dtype_from_torch_name(name: str) -> str | None:
    name = name.lower()
    mapping = {
        "bool": "bool",
        "int8": "int8",
        "int16": "int16",
        "int32": "int32",
        "int64": "int64",
        "uint8": "uint8",
        "uint16": "uint16",
        "uint32": "uint32",
        "uint64": "uint64",
        "float16": "float16",
        "half": "float16",
        "bfloat16": "bfloat16",
        "float32": "float32",
        "float": "float32",
        "float64": "float64",
        "double": "float64",
        "cfloat": "complex",
        "complex64": "complex",
        "complex128": "complex",
    }
    return mapping.get(name)


def dtype_categories(text: str) -> set[str]:
    out: set[str] = set()
    lowered = text.lower()
    macros = {
        "FLOAT_DTYPES": {"float16", "float32", "bfloat16"},
        "ALL_FLOAT_DTYPES": {"float16", "float32", "bfloat16", "float64"},
        "PRIMARY_FLOAT_DTYPES": {"float16", "float32"},
        "INT_DTYPES": {"int16", "int32"},
        "ALL_INT_DTYPES": {"int16", "int32", "int64"},
        "BOOL_TYPES": {"bool"},
        "COMPLEX_DTYPES": {"complex"},
    }
    for macro, dtypes in macros.items():
        if macro in text:
            out.update(dtypes)
    for name in re.findall(r"\b(?:torch|tl)\.([A-Za-z0-9_]+)\b", text):
        dtype = dtype_from_torch_name(name) or DTYPE_CONSTS.get(name)
        if dtype:
            out.add(dtype)
    for name in re.findall(r"['\"]([A-Za-z0-9_]+)['\"]", text):
        dtype = dtype_from_torch_name(name) or DTYPE_CONSTS.get(name)
        if dtype:
            out.add(dtype)
    if "fp8" in lowered or "float8" in lowered:
        out.add("fp8/other")
    return out


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


def int_value_from_node(node: ast.AST, bindings: dict[str, int] | None = None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = int_value_from_node(node.operand, bindings)
        return -value if value is not None else None
    if isinstance(node, ast.Name) and bindings and node.id in bindings:
        return bindings[node.id]
    if isinstance(node, ast.BinOp):
        left = int_value_from_node(node.left, bindings)
        right = int_value_from_node(node.right, bindings)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv) and right:
            return left // right
    return None


def literal_shapes_from_node(node: ast.AST) -> set[str]:
    out: set[str] = set()
    shape = shape_from_literal(node)
    if shape:
        out.add(shape)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for elt in node.elts:
            out.update(literal_shapes_from_node(elt))
    elif isinstance(node, ast.BinOp):
        out.update(literal_shapes_from_node(node.left))
        out.update(literal_shapes_from_node(node.right))
    elif isinstance(node, ast.ListComp):
        out.update(literal_shapes_from_node(node.elt))
        for gen in node.generators:
            out.update(literal_shapes_from_node(gen.iter))
    return out


def target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        out: set[str] = set()
        for elt in node.elts:
            out.update(target_names(elt))
        return out
    return set()


def trim_shape_nw(shape: str) -> str:
    nums = re.findall(r"-?\d+", shape)
    if len(nums) == 4:
        return "(" + ", ".join(nums[:3]) + ")"
    return shape


def parametrize_shape_values(node: ast.AST) -> set[str]:
    if isinstance(node, ast.BinOp):
        return parametrize_shape_values(node.left) | parametrize_shape_values(node.right)
    if isinstance(node, ast.ListComp):
        out: set[str] = set()
        for gen in node.generators:
            names = target_names(gen.target)
            if any("shape" in name.lower() for name in names):
                shapes = set()
                if isinstance(gen.iter, (ast.List, ast.Tuple)) and all(
                    isinstance(elt, ast.Constant) and isinstance(elt.value, int) and not isinstance(elt.value, bool)
                    for elt in gen.iter.elts
                ):
                    shapes = {f"({elt.value},)" for elt in gen.iter.elts if isinstance(elt, ast.Constant)}
                else:
                    shapes = literal_shapes_from_node(gen.iter)
                if any(name.lower() == "shape_nw" for name in names):
                    shapes = {trim_shape_nw(shape) for shape in shapes}
                out.update(shapes)
        return out
    return literal_shapes_from_node(node)


def parametrize_names(node: ast.Call) -> set[str]:
    return set(parametrize_name_list(node))


def parametrize_name_list(node: ast.Call) -> list[str]:
    if not node.args:
        return []
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return [part.strip() for part in first.value.split(",")]
    if isinstance(first, (ast.Tuple, ast.List)):
        names = []
        for elt in first.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                names.append(elt.value)
        return names
    return []


def parametrize_bindings(tree: ast.AST) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not call_name(node.func).endswith("parametrize") or len(node.args) < 2:
            continue
        names = parametrize_name_list(node)
        if not names:
            continue
        values = node.args[1]
        rows = values.elts if isinstance(values, (ast.List, ast.Tuple)) else [values]
        for row in rows:
            if len(names) == 1:
                value = int_value_from_node(row)
                if value is not None:
                    out.append({names[0]: value})
                continue
            if not isinstance(row, (ast.List, ast.Tuple)) or len(row.elts) != len(names):
                continue
            binding: dict[str, int] = {}
            for name, elt in zip(names, row.elts):
                value = int_value_from_node(elt)
                if value is None:
                    break
                binding[name] = value
            if len(binding) == len(names):
                out.append(binding)
    return out


def symbolic_shape_from_node(node: ast.AST, bindings: list[dict[str, int]]) -> set[str]:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return set()
    out: set[str] = set()
    if not bindings:
        bindings = [{}]
    for binding in bindings:
        dims: list[int] = []
        for elt in node.elts:
            value = int_value_from_node(elt, binding)
            if value is None:
                dims = []
                break
            dims.append(value)
        if dims:
            out.add(format_shape(dims))
    return out


def tensor_constructor_shape_args(target: str, node: ast.Call) -> list[ast.AST]:
    base = target.split(".")[-1]
    if base == "randint":
        return [node.args[2]] if len(node.args) >= 3 else []
    if base in {"rand", "randn", "zeros", "ones", "empty", "full", "numpy_random"}:
        return [node.args[0]] if node.args else []
    return []


def call_name(node: ast.AST) -> str:
    chain = attr_chain(node)
    return ".".join(chain) if chain else ""


def shape_samples(text: str) -> set[str]:
    out: set[str] = set()
    for macro, shapes in SHAPE_MACROS.items():
        if macro in text:
            out.update(shapes)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    bindings = parametrize_bindings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = call_name(node.func)
            if target.endswith("parametrize"):
                names = parametrize_names(node)
                relevant = "shape" in names or {"M", "N", "K"} <= names or {"m", "n", "k"} <= names
                if relevant and len(node.args) >= 2:
                    out.update(parametrize_shape_values(node.args[1]))
            for shape_arg in tensor_constructor_shape_args(target, node):
                out.update(literal_shapes_from_node(shape_arg))
                out.update(symbolic_shape_from_node(shape_arg, bindings))
            if target.split(".")[-1] in {"rand", "randn", "randint", "zeros", "ones", "empty", "full", "numpy_random"}:
                for kw in node.keywords:
                    if kw.arg in {"size", "shape"}:
                        out.update(literal_shapes_from_node(kw.value))
                        out.update(symbolic_shape_from_node(kw.value, bindings))
        elif isinstance(node, ast.Assign):
            target_names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if "shape" in target_names:
                out.update(literal_shapes_from_node(node.value))
    return out


def rank_categories(text: str) -> set[str]:
    out: set[str] = set()
    if "REDUCTION_SHAPES" in text or "REDUCTION_SMALL_SHAPES" in text:
        out.update({"2D", "3D"})
    if "POINTWISE_SHAPES" in text:
        out.update({"scalar", "1D", "2D", "3D", "4D", "5D+"})
    if "DISTRIBUTION_SHAPES" in text:
        out.add("3D")
    if re.search(r"broadcast", text, re.IGNORECASE):
        out.add("broadcast")
    if re.search(r"non[_ -]?contiguous|as_strided|stride|transpose|permute|\.t\s*\(", text, re.IGNORECASE):
        out.add("non-contiguous")
    for shape in shape_samples(text):
        rank = shape_rank(shape)
        if rank:
            out.add(rank)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    ignored_parametrize_values: set[int] = set()
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not call_name(call.func).endswith("parametrize") or len(call.args) < 2:
            continue
        names = parametrize_names(call)
        if any("shape" in name.lower() for name in names):
            continue
        for sub in ast.walk(call.args[1]):
            if isinstance(sub, (ast.Tuple, ast.List)):
                ignored_parametrize_values.add(id(sub))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Tuple, ast.List)) and node.elts:
            if id(node) in ignored_parametrize_values:
                continue
            if all(isinstance(e, (ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp)) for e in node.elts):
                if any(isinstance(e, ast.Constant) and isinstance(e.value, int) for e in node.elts):
                    out.add(rank_from_len(len(node.elts)))
    return out


def test_context_from_text(text: str) -> Attrs:
    return Attrs(dtypes=dtype_categories(text), ranks=rank_categories(text), shapes=shape_samples(text))


def dtype_const_from_expr(node: ast.AST, tl_aliases: set[str]) -> set[str]:
    chain = attr_chain(node)
    if chain and chain[0] in tl_aliases:
        dtype = DTYPE_CONSTS.get(chain[-1])
        return {dtype} if dtype else set()
    if chain and chain[0] == "torch":
        dtype = dtype_from_torch_name(chain[-1])
        return {dtype} if dtype else set()
    if isinstance(node, ast.Name):
        dtype = DTYPE_CONSTS.get(node.id)
        return {dtype} if dtype else set()
    return set()


def canonical_tl_op(func: ast.AST, tl_aliases: set[str]) -> str | None:
    chain = attr_chain(func)
    if not chain:
        return None
    name = chain[-1]
    if chain[0] in tl_aliases:
        name = CALL_ALIASES.get(name, name)
        return name if name in CANONICAL_OPS else None
    return None


def canonical_method_op(func: ast.AST) -> str | None:
    chain = attr_chain(func)
    if len(chain) >= 2:
        name = METHOD_ALIASES.get(chain[-1])
        if name in CANONICAL_OPS:
            return name
    if isinstance(func, ast.Attribute):
        name = METHOD_ALIASES.get(func.attr)
        if name in CANONICAL_OPS:
            return name
    return None


def get_shape_rank(node: ast.AST) -> set[str]:
    if isinstance(node, (ast.List, ast.Tuple)):
        return {rank_from_len(len(node.elts))}
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return {"1D"}
    return set()


def slice_rank(base: Attrs, node: ast.Subscript) -> Attrs:
    out = base.copy()
    added = 0
    removed = 0
    slices: list[ast.AST]
    if isinstance(node.slice, ast.Tuple):
        slices = list(node.slice.elts)
    else:
        slices = [node.slice]
    for item in slices:
        if isinstance(item, ast.Constant) and item.value is None:
            added += 1
        elif isinstance(item, ast.Slice):
            continue
        else:
            removed += 1
    if out.ranks:
        new_ranks = set()
        for rank in out.ranks:
            if rank == "scalar":
                n = 0
            elif rank.endswith("D") and rank[:-1].isdigit():
                n = int(rank[:-1])
            else:
                new_ranks.add(rank)
                continue
            new_ranks.add(rank_from_len(max(0, n - removed) + added))
        out.ranks = new_ranks
    elif added:
        out.ranks.add(rank_from_len(added))
    return out


class TritonCallAnalyzer:
    def __init__(
        self,
        path: Path,
        source_kind: str,
        context: Attrs,
        api_ops: set[str] | None = None,
    ) -> None:
        self.path = path
        self.source_kind = source_kind
        self.context = context
        self.api_ops = api_ops or set()
        self.evidence: list[Evidence] = []
        self.tl_aliases: set[str] = {"tl"}
        self.triton_aliases: set[str] = {"triton"}
        self.env: dict[str, Attrs] = {}
        self.current_function = ""

    def analyze_tree(self, tree: ast.AST, only_jit: bool = True) -> list[Evidence]:
        self.tl_aliases, self.triton_aliases = import_aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_jit = any(decorator_is_jit(dec, self.triton_aliases) for dec in node.decorator_list)
                if only_jit and not is_jit:
                    continue
                self.analyze_function(node)
        return self.evidence

    def analyze_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        old_env = self.env
        old_fn = self.current_function
        self.current_function = node.name
        self.env = {}
        for arg in node.args.args + node.args.kwonlyargs:
            self.env[arg.arg] = self.context.copy()
        for stmt in node.body:
            self.process_stmt(stmt)
        self.env = old_env
        self.current_function = old_fn

    def add_evidence(self, op: str, attrs: Attrs, notes: set[str] | None = None) -> None:
        if op not in CANONICAL_OPS:
            return
        ev_notes = set(attrs.notes)
        if notes:
            ev_notes.update(notes)
        ev_shapes = set(attrs.shapes) or set(self.context.shapes)
        self.evidence.append(
            Evidence(
                op=op,
                source_kind=self.source_kind,
                source=rel(self.path),
                function=self.current_function,
                dtypes=set(attrs.dtypes),
                ranks=set(attrs.ranks),
                shapes=ev_shapes,
                api_ops=set(self.api_ops),
                notes=ev_notes,
            )
        )

    def process_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            value = self.infer_expr(stmt.value)
            for target in stmt.targets:
                self.assign_target(target, value)
        elif isinstance(stmt, ast.AnnAssign):
            value = self.infer_expr(stmt.value) if stmt.value else Attrs()
            self.assign_target(stmt.target, value)
        elif isinstance(stmt, ast.AugAssign):
            target = self.infer_expr(stmt.target)
            value = self.infer_expr(stmt.value)
            attrs = Attrs().merge(target).merge(value)
            op = SYNTAX_OPS.get(type(stmt.op))
            if op:
                self.add_evidence(op, attrs, {"tensor/operator syntax evidence"})
            self.assign_target(stmt.target, attrs)
        elif isinstance(stmt, ast.Expr):
            self.infer_expr(stmt.value)
        elif isinstance(stmt, ast.Return):
            if stmt.value:
                self.infer_expr(stmt.value)
        elif isinstance(stmt, ast.If):
            self.process_if(stmt)
        elif isinstance(stmt, (ast.For, ast.While)):
            old = {k: v.copy() for k, v in self.env.items()}
            for sub in stmt.body:
                self.process_stmt(sub)
            body_env = {k: v.copy() for k, v in self.env.items()}
            self.env = old
            for sub in stmt.orelse:
                self.process_stmt(sub)
            self.merge_envs(body_env)
        elif isinstance(stmt, (ast.With, ast.Try)):
            bodies = []
            if isinstance(stmt, ast.With):
                bodies.append(stmt.body)
            else:
                bodies.append(stmt.body)
                for handler in stmt.handlers:
                    bodies.append(handler.body)
                bodies.append(stmt.orelse)
                bodies.append(stmt.finalbody)
            for body in bodies:
                for sub in body:
                    self.process_stmt(sub)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(decorator_is_jit(dec, self.triton_aliases) for dec in stmt.decorator_list):
                self.analyze_function(stmt)

    def process_if(self, stmt: ast.If) -> None:
        old = {k: v.copy() for k, v in self.env.items()}
        for sub in stmt.body:
            self.process_stmt(sub)
        body_env = {k: v.copy() for k, v in self.env.items()}
        self.env = {k: v.copy() for k, v in old.items()}
        for sub in stmt.orelse:
            self.process_stmt(sub)
        else_env = {k: v.copy() for k, v in self.env.items()}
        self.env = {k: v.copy() for k, v in old.items()}
        self.merge_envs(body_env)
        self.merge_envs(else_env)
        self.apply_dtype_if_refinement(stmt, body_env, else_env)

    def apply_dtype_if_refinement(self, stmt: ast.If, body_env: dict[str, Attrs], else_env: dict[str, Attrs]) -> None:
        condition = self.dtype_condition(stmt.test)
        if not condition:
            return
        param, cond_dtypes = condition
        param_dtypes = self.env.get(param, self.context).dtypes or self.context.dtypes
        if not param_dtypes:
            return
        for target in set(body_env) & set(else_env):
            else_attr = else_env[target]
            if "input element dtype" not in else_attr.notes:
                continue
            refined = Attrs()
            refined.merge(body_env[target])
            refined.dtypes.update(param_dtypes - cond_dtypes)
            refined.notes.add(f"conditional accumulator dtype from {param}; {compact(cond_dtypes)} promoted")
            self.env[target] = refined

    def dtype_condition(self, node: ast.AST) -> tuple[str, set[str]] | None:
        if isinstance(node, ast.BoolOp):
            out_param = None
            out_dtypes: set[str] = set()
            for value in node.values:
                item = self.dtype_condition(value)
                if not item:
                    continue
                param, dtypes = item
                out_param = out_param or param
                if param == out_param:
                    out_dtypes.update(dtypes)
            if out_param and out_dtypes:
                return out_param, out_dtypes
        if isinstance(node, ast.Call):
            return self.dtype_condition(node.args[0]) if node.args else None
        if isinstance(node, ast.Compare) and node.ops and isinstance(node.ops[0], ast.Eq):
            left = attr_chain(node.left)
            if len(left) >= 3 and left[-2:] == ["dtype", "element_ty"]:
                dtypes = dtype_const_from_expr(node.comparators[0], self.tl_aliases)
                if dtypes:
                    return left[0], dtypes
        return None

    def merge_envs(self, other: dict[str, Attrs]) -> None:
        for name, attrs in other.items():
            if name in self.env:
                self.env[name].merge(attrs)
            else:
                self.env[name] = attrs.copy()

    def assign_target(self, target: ast.AST, value: Attrs) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = value.copy()
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self.assign_target(elt, value)

    def infer_expr(self, node: ast.AST | None) -> Attrs:
        if node is None:
            return Attrs()
        if isinstance(node, ast.Name):
            return self.env.get(node.id, Attrs()).copy()
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return Attrs({"bool"}, {"scalar"})
            if isinstance(node.value, int):
                return Attrs({"int32"}, {"scalar"})
            if isinstance(node.value, float):
                return Attrs({"float32"}, {"scalar"})
            return Attrs()
        if isinstance(node, ast.Attribute):
            dtypes = dtype_const_from_expr(node, self.tl_aliases)
            if dtypes:
                return Attrs(dtypes=dtypes, notes={"dtype constant"})
            chain = attr_chain(node)
            if len(chain) >= 3 and chain[-2:] == ["dtype", "element_ty"]:
                base = chain[0]
                attrs = self.env.get(base, self.context).copy()
                attrs.notes.add("input element dtype")
                return attrs
            return Attrs()
        if isinstance(node, ast.Call):
            return self.infer_call(node)
        if isinstance(node, ast.BinOp):
            left = self.infer_expr(node.left)
            right = self.infer_expr(node.right)
            attrs = Attrs().merge(left).merge(right)
            op = SYNTAX_OPS.get(type(node.op))
            if op:
                self.add_evidence(op, attrs, {"tensor/operator syntax evidence"})
            return attrs
        if isinstance(node, ast.UnaryOp):
            return self.infer_expr(node.operand)
        if isinstance(node, ast.Compare):
            attrs = Attrs(dtypes={"bool"})
            for child in [node.left, *node.comparators]:
                attrs.ranks.update(self.infer_expr(child).ranks)
            return attrs
        if isinstance(node, ast.BoolOp):
            attrs = Attrs(dtypes={"bool"})
            for value in node.values:
                attrs.ranks.update(self.infer_expr(value).ranks)
            return attrs
        if isinstance(node, ast.IfExp):
            return self.infer_expr(node.body).merge(self.infer_expr(node.orelse))
        if isinstance(node, ast.Subscript):
            return slice_rank(self.infer_expr(node.value), node)
        if isinstance(node, (ast.Tuple, ast.List)):
            attrs = Attrs()
            for elt in node.elts:
                attrs.merge(self.infer_expr(elt))
            return attrs
        return Attrs()

    def infer_call(self, node: ast.Call) -> Attrs:
        arg_attrs = [self.infer_expr(arg) for arg in node.args]
        kw_attrs = {kw.arg: self.infer_expr(kw.value) for kw in node.keywords if kw.arg}
        op = canonical_tl_op(node.func, self.tl_aliases)
        method_op = canonical_method_op(node.func)
        if op:
            attrs = self.attrs_for_tl_call(op, node, arg_attrs, kw_attrs)
            self.add_evidence(op, attrs)
            return attrs
        if method_op:
            attrs = self.attrs_for_method_call(method_op, node, arg_attrs, kw_attrs)
            self.add_evidence(method_op, attrs, {"tensor method evidence"})
            return attrs
        return Attrs().merge_many(arg_attrs) if False else self.merge_attrs_list(arg_attrs + list(kw_attrs.values()))

    def merge_attrs_list(self, items: list[Attrs]) -> Attrs:
        out = Attrs()
        for item in items:
            out.merge(item)
        return out

    def attrs_for_tl_call(self, op: str, node: ast.Call, arg_attrs: list[Attrs], kw_attrs: dict[str, Attrs]) -> Attrs:
        if op in {"program_id", "num_programs"}:
            return Attrs({"int32"}, {"scalar"})
        if op == "arange":
            return Attrs({"int32"}, {"1D"})
        if op in {"zeros", "full"}:
            dtype = kw_attrs.get("dtype", Attrs()).dtypes
            if not dtype and len(node.args) >= 3:
                dtype = self.infer_expr(node.args[2]).dtypes
            rank = get_shape_rank(node.args[0]) if node.args else set()
            return Attrs(set(dtype), set(rank))
        if op == "zeros_like" and arg_attrs:
            return arg_attrs[0].copy()
        if op in {"load", "load_tensor_descriptor"}:
            attrs = Attrs()
            if arg_attrs:
                attrs.dtypes.update(arg_attrs[0].dtypes)
                attrs.ranks.update(arg_attrs[0].ranks)
                attrs.shapes.update(arg_attrs[0].shapes)
            for key in ("mask", "other"):
                attrs.ranks.update(kw_attrs.get(key, Attrs()).ranks)
                attrs.shapes.update(kw_attrs.get(key, Attrs()).shapes)
            attrs.notes.add("load dtype inferred from pointer/test context when available")
            return attrs
        if op in {"store", "store_tensor_descriptor"}:
            value = arg_attrs[1] if len(arg_attrs) > 1 else Attrs()
            attrs = value.copy()
            attrs.notes.add("store evidence uses stored value dtype/rank")
            return attrs
        if op == "cast":
            target_dtype = set()
            if len(node.args) >= 2:
                target_dtype.update(self.infer_expr(node.args[1]).dtypes)
            target_dtype.update(kw_attrs.get("dtype", Attrs()).dtypes)
            base = arg_attrs[0].copy() if arg_attrs else Attrs()
            if target_dtype:
                base.dtypes = target_dtype
                base.notes.add(f"converted via cast/to to {compact(target_dtype)}")
            return base
        if op in REDUCTION_OPS:
            attrs = arg_attrs[0].copy() if arg_attrs else Attrs()
            axis_text = ""
            if len(node.args) >= 2:
                axis_text = ast.unparse(node.args[1])
            for kw in node.keywords:
                if kw.arg in {"axis", "keep_dims"}:
                    axis_text += f" {kw.arg}={ast.unparse(kw.value)}"
            if axis_text:
                attrs.notes.add(f"axis/keep_dims: {axis_text.strip()}")
            return attrs
        if op == "dot":
            attrs = self.merge_attrs_list(arg_attrs[:2])
            attrs.ranks = {"2D"}
            return attrs
        if op == "dot_scaled":
            attrs = self.merge_attrs_list(arg_attrs[:6])
            attrs.ranks = {"2D"}
            attrs.notes.add("scaled dot operand evidence")
            return attrs
        if op in {"rand", "rand4x", "randn", "randn4x"}:
            attrs = Attrs({"float32"})
            if len(arg_attrs) >= 2:
                attrs.ranks.update(arg_attrs[1].ranks)
            return attrs
        if op in {"randint", "randint4x"}:
            attrs = Attrs({"int32"})
            if len(arg_attrs) >= 2:
                attrs.ranks.update(arg_attrs[1].ranks)
            return attrs
        if op in {"where"}:
            attrs = self.merge_attrs_list(arg_attrs[1:3] if len(arg_attrs) >= 3 else arg_attrs)
            if arg_attrs:
                attrs.ranks.update(arg_attrs[0].ranks)
            return attrs
        if op in ELEMENTWISE_ARG_OPS or op.startswith("atomic_"):
            return self.merge_attrs_list(arg_attrs)
        return self.merge_attrs_list(arg_attrs + list(kw_attrs.values()))

    def attrs_for_method_call(self, op: str, node: ast.Call, arg_attrs: list[Attrs], kw_attrs: dict[str, Attrs]) -> Attrs:
        chain = attr_chain(node.func)
        base = Attrs()
        if chain:
            # Reconstruct method base by walking the value expression.
            func = node.func
            if isinstance(func, ast.Attribute):
                base = self.infer_expr(func.value)
        if op == "cast":
            target_dtype = set()
            if node.args:
                target_dtype.update(self.infer_expr(node.args[0]).dtypes)
            target_dtype.update(kw_attrs.get("dtype", Attrs()).dtypes)
            if target_dtype:
                base.dtypes = target_dtype
                base.notes.add(f"converted via .to/.cast to {compact(target_dtype)}")
            return base
        elif op in {"reshape", "view", "broadcast_to"}:
            if node.args:
                base.ranks = {rank_from_len(len(node.args))}
        elif op in {"trans", "permute"}:
            base.notes.add("shape permutation before/at Triton op")
            base.ranks.add("non-contiguous")
        return base


def line_context(text: str, node: ast.AST) -> str:
    if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
        lines = text.splitlines()
        start = node.lineno
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start = min(getattr(dec, "lineno", start) for dec in decorators)
        return "\n".join(lines[start - 1 : node.end_lineno])
    return text


def regex_tl_evidence(path: Path, text: str, source_kind: str, context: Attrs, api_ops: set[str], function: str) -> list[Evidence]:
    out: list[Evidence] = []
    for match in re.finditer(r"\btl(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", text):
        name = match.group(0).split(".")[-1]
        op = CALL_ALIASES.get(name, name)
        if op not in CANONICAL_OPS:
            continue
        out.append(
            Evidence(
                op=op,
                source_kind=source_kind,
                source=rel(path),
                function=function,
                dtypes=set(),
                ranks=set(),
                shapes=set(context.shapes),
                api_ops=set(api_ops),
                notes={"text/template evidence; dtype/rank not inferred at Triton operand boundary"},
            )
        )
    return out


def atomic_rmw_dtypes_by_op(text: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Tuple, ast.List)) or len(node.elts) < 2:
            continue
        op_node, dtype_node = node.elts[:2]
        if not (
            isinstance(op_node, ast.Constant)
            and isinstance(op_node.value, str)
            and isinstance(dtype_node, ast.Constant)
            and isinstance(dtype_node.value, str)
        ):
            continue
        op = f"atomic_{op_node.value}"
        if op not in CANONICAL_OPS:
            continue
        dtype = dtype_from_torch_name(dtype_node.value) or DTYPE_CONSTS.get(dtype_node.value)
        if dtype:
            out[op].add(dtype)
    return out


def reduce_test_attrs_for_op(op: str) -> Attrs:
    if op == "xor_sum":
        return Attrs(
            dtypes={"bool"},
            ranks={"2D"},
            shapes=set(SHAPE_MACROS["reduce_bool"]),
            notes={"dynamic patched-kernel evidence; xor_sum uses reduce_bool configs"},
        )

    shapes = {"(1, 1024)", *SHAPE_MACROS["reduce_configs3"], "(32, 32)", "(32, 2, 16)"}
    if op in {"min", "max", "sum"}:
        shapes.update(SHAPE_MACROS["reduce_configs2"])
    elif op in {"argmin", "argmax"}:
        shapes.update({"(2, 32)", "(4, 32)", "(4, 128)"})
    ranks = {rank for shape in shapes for rank in [shape_rank(shape)] if rank}
    return Attrs(
        dtypes=set(DTYPES_WITH_BFLOAT16),
        ranks=ranks,
        shapes=shapes,
        notes={"dynamic patched-kernel evidence; reduce op-specific parametrization"},
    )


def dynamic_direct_evidence(path: Path, text: str, context: Attrs, function: str, dedicated_ops: set[str]) -> list[Evidence]:
    out: list[Evidence] = []
    special_dynamic = {
        "test_math_divide_op": ("math divide expr parametrized", {"float32"}, {"1D"}, {"(128,)"}),
        "test_precise_math": ("precise math expr parametrized", {"float32"}, {"1D"}, {"(128,)"}),
        "test_unary_math": ("unary math function parametrized", {"float32"}, {"1D"}, {"(128,)"}),
    }
    if function in special_dynamic:
        note, dtypes, ranks, shapes = special_dynamic[function]
        out.extend(
            Evidence(
                op=op,
                source_kind="direct-triton-test",
                source=rel(path),
                function=function,
                dtypes=set(dtypes),
                ranks=set(ranks),
                shapes=set(shapes),
                notes={f"dynamic patched-kernel evidence; {note}"},
            )
            for op in sorted(dedicated_ops)
        )
    if "tl.atomic_{op}" in text:
        dtypes_by_op = atomic_rmw_dtypes_by_op(text)
        out.extend(
            Evidence(
                op=op,
                source_kind="direct-triton-test",
                source=rel(path),
                function=function,
                dtypes=set(dtypes_by_op.get(op, context.dtypes)),
                ranks={"1D"} if function == "test_atomic_rmw" else set(context.ranks),
                shapes={"(5,)"} if function == "test_atomic_rmw" else set(context.shapes),
                notes={"dynamic patched-kernel evidence; dtype inferred from op-specific parametrization"},
            )
            for op in sorted(dedicated_ops)
            if op.startswith("atomic_")
        )
    has_dynamic_tl_op = "tl.{op}" in text or "tl.{op." in text or "tl.{expr}" in text
    if has_dynamic_tl_op:
        for op in sorted(dedicated_ops):
            if function == "test_math_op":
                attrs = Attrs(
                    dtypes={"float32", "float64"},
                    ranks={"1D"},
                    shapes={"(128,)"},
                    notes={"dynamic patched-kernel evidence; math expr parametrized"},
                )
            elif function == "test_reduce1d":
                attrs = Attrs(
                    dtypes=set(DTYPES_WITH_BFLOAT16),
                    ranks={"1D"},
                    shapes={"(32,)", "(64,)", "(128,)", "(512,)"},
                    notes={"dynamic patched-kernel evidence; reduce1d op parametrized"},
                )
            elif function == "test_scan2d":
                attrs = Attrs(
                    dtypes={"int32", "float32", "bfloat16"},
                    ranks={"2D"},
                    shapes=set(SHAPE_MACROS["scan_configs"]),
                    notes={"dynamic patched-kernel evidence; scan op parametrized"},
                )
            elif function == "test_reduce" and op in {"argmax", "argmin", "max", "min", "sum", "xor_sum"}:
                attrs = reduce_test_attrs_for_op(op)
            else:
                attrs = context.copy()
                attrs.notes.add("dynamic patched-kernel evidence; dtype/rank inferred from test context")
            out.append(
                Evidence(
                    op=op,
                    source_kind="direct-triton-test",
                    source=rel(path),
                    function=function,
                    dtypes=set(attrs.dtypes),
                    ranks=set(attrs.ranks),
                    shapes=set(attrs.shapes),
                    notes=set(attrs.notes),
                )
            )
    return out


def direct_sources() -> list[Path]:
    paths: list[Path] = []
    for root in DIRECT_TEST_ROOTS:
        if root.exists():
            paths.extend(
                p
                for p in sorted(root.rglob("test_*.py"))
                if p.is_file() and not (EXCLUDE_PARTS & set(p.parts))
            )
    return paths


def static_tl_ops_from_source(path: Path, src: str) -> set[str]:
    ops: set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = None
    if tree is not None:
        context = test_context_from_text(src)
        analyzer = TritonCallAnalyzer(path, "direct-triton-test", context)
        ops.update(ev.op for ev in analyzer.analyze_tree(tree, only_jit=True))
    for ev in regex_tl_evidence(path, src, "direct-triton-test", Attrs(), set(), ""):
        ops.add(ev.op)
    return ops


def test_mapping_key(source: str, function: str) -> tuple[str, str]:
    return (source, function)


def direct_test_ops(node: ast.FunctionDef | ast.AsyncFunctionDef, src: str, path: Path) -> set[str]:
    if node.name in DIRECT_TEST_EXCLUDES:
        return set()
    if node.name in DIRECT_TEST_OP_OVERRIDES:
        return set(DIRECT_TEST_OP_OVERRIDES[node.name])
    name_ops = dedicated_ops_from_direct_context(node.name, src, path)
    return name_ops & static_tl_ops_from_source(path, src)


def collect_direct_test_mappings() -> list[TestMapping]:
    mappings: set[TestMapping] = set()
    for path in direct_sources():
        text = read_text(path)
        tree = parse_file(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            src = line_context(text, node)
            ops = direct_test_ops(node, src, path)
            if not ops:
                continue
            reason = "direct test name"
            if node.name in DIRECT_TEST_OP_OVERRIDES:
                reason = "direct test family override"
            for op in ops:
                mappings.add(
                    TestMapping(
                        op=op,
                        source_kind="direct-triton-test",
                        source=rel(path),
                        function=node.name,
                        reason=reason,
                    )
                )
    return sorted(mappings, key=lambda item: (item.op, item.source_kind, item.source, item.function, item.api))


def collect_direct_evidence(mappings: list[TestMapping] | None = None) -> list[Evidence]:
    evidence: list[Evidence] = []
    mappings = mappings if mappings is not None else collect_direct_test_mappings()
    ops_by_test: dict[tuple[str, str], set[str]] = defaultdict(set)
    for mapping in mappings:
        if mapping.source_kind != "direct-triton-test":
            continue
        ops_by_test[test_mapping_key(mapping.source, mapping.function)].add(mapping.op)
    for path in direct_sources():
        text = read_text(path)
        tree = parse_file(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            src = line_context(text, node)
            context = test_context_from_text(src)
            dedicated_ops = ops_by_test.get(test_mapping_key(rel(path), node.name), set())
            if not dedicated_ops:
                continue
            sub_tree = None
            try:
                sub_tree = ast.parse(src)
            except SyntaxError:
                pass
            if sub_tree is not None:
                analyzer = TritonCallAnalyzer(path, "direct-triton-test", context)
                evidence.extend(ev for ev in analyzer.analyze_tree(sub_tree, only_jit=True) if ev.op in dedicated_ops)
            evidence.extend(
                ev for ev in regex_tl_evidence(path, src, "direct-triton-test", context, set(), node.name)
                if ev.op in dedicated_ops
            )
            evidence.extend(dynamic_direct_evidence(path, src, context, node.name, dedicated_ops))
    return evidence


def valid_flaggems_source(path: Path) -> bool:
    parts = set(path.parts)
    if EXCLUDE_PARTS & parts:
        return False
    if "runtime" in parts and "_kunpeng" not in parts:
        return False
    return path.suffix == ".py"


def flaggems_source_files() -> list[Path]:
    paths: list[Path] = []
    for root in FLAGGEMS_ROOTS:
        if root.exists():
            paths.extend(p for p in root.rglob("*.py") if valid_flaggems_source(p))
    return sorted(set(paths))


def public_defs(path: Path) -> set[str]:
    tree = parse_file(path)
    if tree is None:
        return {path.stem}
    names = {path.stem}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def build_source_index(paths: list[Path]) -> dict[str, set[Path]]:
    by_name: dict[str, set[Path]] = defaultdict(set)
    for path in paths:
        for name in public_defs(path):
            by_name[name].add(path)
    return by_name


def marker_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        chain = attr_chain(target)
        if len(chain) >= 3 and chain[0] == "pytest" and chain[1] == "mark":
            names.add(chain[2])
    return names


def api_names_from_text(text: str) -> set[str]:
    names = set(re.findall(r"\b(?:flag_gems|gems)\.([A-Za-z_][A-Za-z0-9_]*)\b", text))
    return names


def collect_flaggems_contexts(source_names: set[str]) -> dict[str, dict[str, object]]:
    contexts: dict[str, dict[str, object]] = defaultdict(lambda: {"attrs": Attrs(), "tests": set()})
    tests_root = ROOT / "FlagGems/tests"
    for path in sorted(tests_root.glob("**/*.py")):
        if EXCLUDE_PARTS & set(path.parts):
            continue
        text = read_text(path)
        tree = parse_file(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            src = line_context(text, node)
            markers = marker_names(node) & source_names
            name_matches = dedicated_names_from_parts(
                [node.name, path.stem.removeprefix("test_")],
                source_names,
            )
            names = markers | name_matches
            if not names:
                continue
            attrs = test_context_from_text(src)
            label = f"{rel(path)}::{node.name}"
            for name in names:
                contexts[name]["attrs"].merge(attrs)  # type: ignore[index,union-attr]
                contexts[name]["tests"].add(label)  # type: ignore[index,union-attr]
    return contexts


def collect_flaggems_test_mappings(contexts: dict[str, dict[str, object]]) -> list[TestMapping]:
    mappings: set[TestMapping] = set()
    for api, ctx in contexts.items():
        dedicated_ops = dedicated_triton_ops_for_api(api)
        if not dedicated_ops:
            continue
        tests = ctx["tests"]  # type: ignore[index]
        for label in tests:
            source, _, function = str(label).partition("::")
            for op in dedicated_ops:
                mappings.add(
                    TestMapping(
                        op=op,
                        source_kind="flaggems-api-test",
                        source=source,
                        function=function,
                        api=api,
                        reason="FlagGems marker/test name",
                    )
                )
    return sorted(mappings, key=lambda item: (item.op, item.source_kind, item.source, item.function, item.api))


def collect_flaggems_evidence(
    contexts: dict[str, dict[str, object]] | None = None,
    mappings: list[TestMapping] | None = None,
) -> list[Evidence]:
    evidence: list[Evidence] = []
    paths = flaggems_source_files()
    by_name = build_source_index(paths)
    contexts = contexts if contexts is not None else collect_flaggems_contexts(set(by_name))
    mappings = mappings if mappings is not None else collect_flaggems_test_mappings(contexts)
    mapped_api_to_ops: dict[str, set[str]] = defaultdict(set)
    for mapping in mappings:
        if mapping.source_kind == "flaggems-api-test" and mapping.api:
            mapped_api_to_ops[mapping.api].add(mapping.op)
    selected: dict[Path, set[str]] = defaultdict(set)
    for api, ctx in contexts.items():
        for path in by_name.get(api, set()):
            if path.name != "__init__.py":
                selected[path].add(api)
    for path, api_ops in selected.items():
        tree = parse_file(path)
        if tree is None:
            continue
        context = Attrs()
        api_to_labels: dict[str, str] = {}
        api_to_ops: dict[str, set[str]] = {}
        for api in api_ops:
            dedicated_ops = mapped_api_to_ops.get(api, set())
            if not dedicated_ops:
                continue
            context.merge(contexts[api]["attrs"])  # type: ignore[arg-type]
            tests = contexts[api]["tests"]  # type: ignore[index]
            api_to_labels[api] = f"{api} ({compact(tests, 3)})"
            api_to_ops[api] = dedicated_ops
        if not api_to_ops:
            continue
        analyzer = TritonCallAnalyzer(path, "flaggems-triton-kernel", context, set(api_to_ops))
        evs = analyzer.analyze_tree(tree, only_jit=True)
        for ev in evs:
            labels = {api_to_labels[api] for api, ops in api_to_ops.items() if ev.op in ops}
            if not labels:
                continue
            ev.api_ops = labels
            evidence.append(ev)
    return evidence


def language_notes() -> dict[str, set[str]]:
    notes: dict[str, set[str]] = defaultdict(set)
    for path in [
        ROOT / "python/triton/language/standard.py",
        ROOT / "python/triton/language/core.py",
        ROOT / "python/triton/language/math.py",
        ROOT / "python/triton/language/random.py",
    ]:
        tree = parse_file(path)
        if tree is None:
            continue
        text = read_text(path)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = CALL_ALIASES.get(node.name, node.name)
            if name not in CANONICAL_OPS:
                continue
            src = line_context(text, node)
            if "_promote_bfloat16_to_float32" in src:
                notes[name].add("Triton language implementation promotes bfloat16 to float32")
            if "dtype.element_ty == tl.float16" in src or "dtype.element_ty == tl.bfloat16" in src:
                notes[name].add("implementation contains dtype-dependent accumulator promotion")
    return notes


def build_rows(evidence: list[Evidence]) -> list[Row]:
    rows = {op: Row(OP_TO_GROUP[op], op) for op in CANONICAL_OPS}
    lang_notes = language_notes()
    for op, notes in lang_notes.items():
        rows[op].notes.update(notes)
    for ev in evidence:
        row = rows.get(ev.op)
        if row is None:
            continue
        row.evidence_count += 1
        row.dtypes.update(ev.dtypes)
        row.ranks.update(ev.ranks & set(RANK_COLS))
        row.shapes.update(ev.shapes)
        if ev.source_kind == "direct-triton-test":
            row.direct.add(f"{ev.source}::{ev.function}" if ev.function else ev.source)
        else:
            row.flaggems.update(ev.api_ops or {ev.source})
        row.notes.update(ev.notes)
    return sorted(rows.values(), key=lambda r: (GROUP_ORDER[r.group], OP_ORDER[r.op], r.op))


def markdown_table(rows: list[Row]) -> str:
    headers = [
        "Group",
        "Triton op",
        *DTYPE_COLS,
        *RANK_COLS,
        "Evidence count",
        "Direct Triton tests",
        "Notes",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [
            row.group,
            row.op,
            *["Y" if col in row.dtypes else "" for col in DTYPE_COLS],
            *["Y" if col in row.ranks else "" for col in RANK_COLS],
            str(row.evidence_count),
            summarize_direct_evidence(row.direct),
            compact_notes(row.notes),
        ]
        safe = [v.replace("|", "\\|").replace("\n", " ") for v in values]
        lines.append("| " + " | ".join(safe) + " |")
    return "\n".join(lines)


def output_dtype_supported(row: Row, col: str) -> bool:
    return bool(row.dtypes & OUTPUT_DTYPE_ALIASES[col])


def compact_ranks(ranks: set[str]) -> str:
    ordered = [rank for rank in RANK_COLS if rank in ranks]
    return ";".join(ordered)


def display_ranks(row: Row) -> set[str]:
    if row.op in {"dot", "dot_scaled"}:
        return {"2D"}
    ranks = {rank for shape in row.shapes for rank in [shape_rank(shape)] if rank}
    if ranks:
        ranks.update(row.ranks & {"broadcast", "non-contiguous"})
        return ranks
    return set(row.ranks)


def markdown_mapping_table(mappings: list[TestMapping]) -> str:
    by_op: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"direct": set(), "flaggems": set(), "reasons": set()})
    for mapping in mappings:
        item = f"{mapping.source}::{mapping.function}" if mapping.function else mapping.source
        if mapping.api:
            item = f"{mapping.api}: {item}"
        key = "direct" if mapping.source_kind == "direct-triton-test" else "flaggems"
        by_op[mapping.op][key].add(item)
        if mapping.reason:
            by_op[mapping.op]["reasons"].add(mapping.reason)

    headers = ["Group", "Triton op", "Direct Triton tests", "API tests", "Match rule"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for op in sorted(CANONICAL_OPS, key=lambda item: (GROUP_ORDER[OP_TO_GROUP[item]], OP_ORDER[item], item)):
        data = by_op.get(op, {"direct": set(), "flaggems": set(), "reasons": set()})
        values = [
            OP_TO_GROUP[op],
            op,
            compact(data["direct"], 8),
            compact(data["flaggems"], 8),
            compact(data["reasons"], 4),
        ]
        safe = [v.replace("|", "\\|").replace("\n", " ") for v in values]
        lines.append("| " + " | ".join(safe) + " |")
    return "\n".join(lines)


def write_mapping_outputs(mappings: list[TestMapping]) -> None:
    summary = (
        "# Triton Operator Test Mapping\n\n"
        "Scope: dedicated test mapping used by the support matrix generator. "
        "Evidence extraction only counts `tl.*` calls for ops listed here for each test/API.\n\n"
        f"- mapped records: {len(mappings)}\n\n"
    )
    (OUT_DIR / "triton_op_test_mapping.md").write_text(
        summary + markdown_mapping_table(mappings) + "\n",
        encoding="utf-8",
    )
    with (OUT_DIR / "triton_op_test_mapping.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Triton op", "Source kind", "Test source", "Test function", "API", "Match rule"])
        for mapping in mappings:
            writer.writerow(
                [
                    mapping.op,
                    mapping.source_kind,
                    mapping.source,
                    mapping.function,
                    mapping.api,
                    mapping.reason,
                ]
            )


def write_outputs(rows: list[Row], evidence: list[Evidence], mappings: list[TestMapping]) -> None:
    summary = (
        "# Triton-Level Operator Support Matrix\n\n"
        "Scope: user-provided Triton operator list. Every listed operator is emitted as a row.\n\n"
        "Evidence scope: direct Triton tests under `python/` only. FlagGems-derived evidence is excluded "
        "from this Triton-op matrix.\n\n"
        "Dedicated test mapping is written to `/tmp/triton_op_test_mapping.md` and "
        "`/tmp/triton_op_test_mapping.csv`. Evidence extraction is restricted by that mapping.\n\n"
        "Important: dtype columns are Triton-level evidence. Dimension coverage is derived from concrete "
        "`Shape Size` when available, with layout markers such as broadcast/non-contiguous preserved.\n\n"
        f"- operators in reference list: {len(rows)}\n"
        f"- test mapping records: {len(mappings)}\n"
        f"- evidence records: {len(evidence)}\n"
        f"- direct Triton sources scanned: {len(direct_sources())}\n"
    )
    write_mapping_outputs(mappings)
    md = summary + markdown_table(rows) + "\n"
    (OUT_DIR / "triton_level_support_matrix.md").write_text(md, encoding="utf-8")

    headers = ["分类", "算子", *OUTPUT_DTYPE_COLS, "不支持原因", "维度覆盖", "Shape Size"]
    with (OUT_DIR / "triton_level_support_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(
                [
                    row.group,
                    row.op,
                    *["Y" if output_dtype_supported(row, col) else "" for col in OUTPUT_DTYPE_COLS],
                    "",
                    compact_ranks(display_ranks(row)),
                    compact(row.shapes, 100),
                ]
            )

    with (OUT_DIR / "triton_level_support_evidence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Triton op", "Source kind", "Source", "Function", "Actual dtypes", "Actual ranks", "Actual shapes", "API evidence", "Notes"])
        for ev in evidence:
            writer.writerow(
                [
                    ev.op,
                    ev.source_kind,
                    ev.source,
                    ev.function,
                    compact(ev.dtypes, 100),
                    compact(ev.ranks, 100),
                    compact(ev.shapes, 100),
                    compact(ev.api_ops, 100),
                    compact(ev.notes, 100),
                ]
            )


def main() -> None:
    mappings = collect_direct_test_mappings()

    evidence = collect_direct_evidence(mappings)
    rows = build_rows(evidence)
    write_outputs(rows, evidence, mappings)
    covered = sum(1 for row in rows if row.evidence_count)
    print(f"rows={len(rows)}")
    print(f"covered_rows={covered}")
    print(f"test_mapping_records={len(mappings)}")
    print(f"evidence_records={len(evidence)}")
    print(OUT_DIR / "triton_op_test_mapping.md")
    print(OUT_DIR / "triton_op_test_mapping.csv")
    print(OUT_DIR / "triton_level_support_matrix.md")
    print(OUT_DIR / "triton_level_support_matrix.csv")
    print(OUT_DIR / "triton_level_support_evidence.csv")


if __name__ == "__main__":
    main()
