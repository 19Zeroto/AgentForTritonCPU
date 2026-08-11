#!/usr/bin/env python3
"""Build the fixed metric map and export reports from FlagGems record logs.

Usage: ``python3 fusion_metrics.py report --logs <run-dir>``.  The map defaults
to ``fusion_metrics_map.json`` next to this script, and the report defaults to
``<run-dir>/report``.  ``build-map`` materializes logical FLOPs/bytes;
``update-report`` replaces selected operators.  The calculator deliberately
does not import torch, FlagGems, or execute a benchmark.

Design: ``../references/design.md``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

try:
    import yaml
except ImportError as exc:  # pragma: no cover - deployment error
    raise SystemExit("PyYAML is required by fusion_metrics.py") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
FORMULA_PATH = SCRIPT_DIR / "fusion_metrics_formulas.yaml"
MAP_PATH = SCRIPT_DIR / "fusion_metrics_map.json"
OP_ALIASES = {
    "skip_layernorm": "skip_layer_norm",
    "weight_norm_interface": "weight_norm",
}

COMPUTE_HEADER = [
    "算子名称", "shape", "dtype", "Gems延迟(ms)", "Torch延迟(ms)",
    "目标算力下期望延迟(ms)", "目标达成率(%)", "逻辑计算量(GFLOPs)",
    "实测算力(TFLOPS)", "目标算力(TFLOPS)", "实测/期望延迟倍率",
    "Torch Speedup", "源码文件", "pipeline", "OMP线程数", "warmup", "iter",
    "benchmark模式", "备注",
]
MEMORY_HEADER = [
    "算子名称", "shape", "dtype", "Gems延迟(ms)", "Torch延迟(ms)",
    "目标带宽下期望延迟(ms)", "目标达成率(%)", "逻辑数据量(GB)",
    "逻辑带宽(GB/s)", "目标带宽(GB/s)", "Torch逻辑带宽(GB/s)", "Speedup",
    "源码文件", "pipeline", "OMP线程数", "warmup", "iter", "benchmark模式", "备注",
]
COVERAGE_HEADER = [
    "算子名称", "类别", "状态", "原因", "测试文件", "marker", "pipeline",
    "NUMA节点", "stdout", "record log",
]

TARGET_TFLOPS = {
    ("SME", "float32"): 2.52631579,
    ("SME", "bfloat16"): 5.05263158,
    ("SVE", "float32"): 0.71578947,
}
DEFAULT_PIPELINE_BY_CATEGORY = {
    "compute": "SME",
    "memory": "SVE",
}
SOURCE_FILES = {
    "silu_and_mul": "fused/silu_and_mul.py",
    "silu_and_mul_out": "fused/silu_and_mul.py",
    "gelu_and_mul": "fused/gelu_and_mul.py",
    "geglu": "fused/geglu.py",
    "reglu": "fused/reglu.py",
    "skip_layer_norm": "fused/skip_layernorm.py",
    "fused_add_rms_norm": "fused/fused_add_rms_norm.py",
    "apply_rotary_pos_emb": "fused/rotary_embedding.py",
    "apply_repetition_penalties": "fused/apply_repetition_penalties.py",
    "cross_entropy_loss": "fused/cross_entropy_loss.py",
    "concat_and_cache_mla": "fused/concat_and_cache_mla.py",
    "reshape_and_cache": "fused/reshape_and_cache.py",
    "reshape_and_cache_flash": "fused/reshape_and_cache_flash.py",
}


class FusionMetricError(ValueError):
    """Invalid formula, log, or mapping input."""


@dataclass(frozen=True)
class TensorMeta:
    shape: tuple[int, ...]
    element_size: int
    values: tuple[int, ...] | None = None

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1


@dataclass(frozen=True)
class Formula:
    name: str
    category: str
    metric: str
    inputs: Mapping[str, Mapping[str, Any]]
    expression: str
    enabled: bool
    aliases: tuple[str, ...]
    tree: ast.Expression


def canonical_op(name: str) -> str:
    name = name.removeprefix("flag_gems.")
    return OP_ALIASES.get(name, name)


def canonical_dtype(value: Any) -> str:
    text = str(value).removeprefix("torch.").lower()
    aliases = {
        "fp32": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
        "half": "float16",
    }
    return aliases.get(text, text)


def normalize(value: Any) -> Any:
    if isinstance(value, tuple):
        return [normalize(item) for item in value]
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize(value[key]) for key in sorted(value)}
    return value


def make_key(op_name: str, dtype: Any, shape_detail: Any) -> str:
    return json.dumps(
        [canonical_op(op_name), canonical_dtype(dtype), normalize(shape_detail)],
        separators=(",", ":"),
        ensure_ascii=False,
    )


def load_formulas(path: Path = FORMULA_PATH) -> dict[str, Formula]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("formulas"), dict):
        raise FusionMetricError(f"invalid formula file: {path}")
    result: dict[str, Formula] = {}
    aliases: dict[str, str] = {}
    for name, entry in raw["formulas"].items():
        if not isinstance(entry, dict):
            raise FusionMetricError(f"formula {name} is not a mapping")
        expression = str(entry.get("expression", ""))
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.Attribute,
                    ast.Lambda,
                    ast.comprehension,
                    ast.DictComp,
                    ast.ListComp,
                    ast.Call,
                ),
            ):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    continue
                raise FusionMetricError(
                    f"unsupported expression in {name}: {type(node).__name__}"
                )
        formula = Formula(
            name,
            entry.get("category", ""),
            entry.get("metric", ""),
            entry.get("inputs", {}),
            expression,
            bool(entry.get("enabled", True)),
            tuple(entry.get("aliases", [])),
            tree,
        )
        result[name] = formula
        for alias in formula.aliases:
            aliases[alias] = name
    for alias, target in aliases.items():
        result[alias] = result[target]
    return result


def _numel(value: Any) -> int:
    if isinstance(value, TensorMeta):
        return value.numel()
    if isinstance(value, (list, tuple)):
        return math.prod(int(item) for item in value) if value else 1
    return int(value)


def _nbytes(value: Any) -> int:
    if value is None:
        return 0
    if not isinstance(value, TensorMeta):
        return 0
    return value.numel() * value.element_size


def _dim(value: Any, index: Any) -> int:
    if not isinstance(value, TensorMeta):
        raise FusionMetricError("dim() expects tensor metadata")
    return int(value.shape[int(index)])


def _shape_numel(value: Any) -> int:
    return _numel(value)


def _tensor_sum(value: Any) -> int:
    if isinstance(value, TensorMeta) and value.values is not None:
        return sum(value.values)
    if isinstance(value, (list, tuple)):
        return sum(int(item) for item in value)
    raise FusionMetricError("fixed sequence values are required for tensor_sum")


def _attention_pair_count(
    query_length: Any,
    key_length: Any,
    causal: Any,
    left: Any,
    right: Any,
) -> int:
    q_len = int(query_length)
    kv_len = int(key_length)
    left = int(left)
    right = int(right)
    alignment = (
        kv_len - q_len
        if bool(causal) or (left >= 0 and right >= 0)
        else 0
    )
    pairs = 0
    for query_index in range(q_len):
        center = query_index + alignment
        start = 0 if left < 0 else max(0, center - left)
        end = min(kv_len, center + 1) if causal else kv_len
        if right >= 0:
            end = min(end, center + right + 1)
        pairs += max(0, end - start)
    return pairs


_VARLEN_WORKLOADS = {
    # Shape-only logs retain [length] for these tensors, not their values.
    (512, 2, 1): ((0, 512), (512,)),
    (72, 4, 3): ((0, 1, 2, 72), (1, 1, 70)),
    (265, 56, 55): (
        tuple(range(0, 45))
        + (105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265),
        (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
    ),
    (265, 201, 200): (
        tuple(range(0, 196)) + (211, 226, 240, 253, 265),
        (2333,)
        + (2331,) * 20
        + (2330,) * 20
        + (2329,) * 14
        + (2328,) * 18
        + (2327,) * 15
        + (2326,) * 17
        + (2325,) * 18
        + (2324,) * 21
        + (2323,) * 22
        + (2322,) * 24
        + (2321,) * 5
        + (2320, 2319, 2318, 2317, 2316),
    ),
}


def _varlen_attention_flops(
    query: TensorMeta,
    cumulative: Any,
    used: Any,
    causal: Any,
    window: Any,
) -> int:
    if isinstance(cumulative, TensorMeta):
        cumulative = list(cumulative.shape)
    if isinstance(used, TensorMeta):
        used = list(used.shape)
    if not isinstance(cumulative, (list, tuple)) or not isinstance(used, (list, tuple)):
        raise FusionMetricError(
            "varlen workload lengths are not present in shape detail"
        )
    # If a future logger preserves values, accept those directly; otherwise
    # resolve the fixed benchmark workload from recorded query/vector lengths.
    if len(cumulative) == 1 and len(used) == 1:
        key = (int(query.shape[0]), int(cumulative[0]), int(used[0]))
        workload = _VARLEN_WORKLOADS.get(key)
        if workload is None:
            raise FusionMetricError(f"unknown fixed varlen workload {key}")
        cumulative_values, used_values = workload
    elif len(cumulative) == len(used) + 1:
        cumulative_values, used_values = cumulative, used
    else:
        raise FusionMetricError(
            "cumulative query lengths must have batch_size + 1 entries"
        )
    left, right = (-1, -1) if window is None else window
    pairs = sum(
        _attention_pair_count(
            int(cumulative_values[i + 1]) - int(cumulative_values[i]),
            int(used_values[i]),
            causal,
            left,
            right,
        )
        for i in range(len(used_values))
    )
    return 4 * _dim(query, 1) * _dim(query, 2) * pairs


def _flash_mla_flops(
    max_seqlen_pad: Any,
    batch_size: Any,
    query_length: Any,
    query_heads: Any,
    query_key_dim: Any,
    value_dim: Any,
) -> int:
    batch = int(batch_size)
    # The benchmark uses sequence lengths [1024, 2048, ...] and rounds
    # max(seq + 2 * (B - 1)) to the next 256-token boundary.  For B=128 this
    # fixed workload rule is exactly max_seqlen_pad - 256.
    base = int(max_seqlen_pad) - 256
    if batch <= 0 or base < 0:
        raise FusionMetricError("invalid fixed flash_mla workload")
    sequence_sum = batch * base + batch * (batch - 1)
    return (
        2
        * int(query_length)
        * int(query_heads)
        * sequence_sum
        * (int(query_key_dim) + int(value_dim))
    )


def _sparse_mla_flops(query: TensorMeta, indices: TensorMeta, value_dim: Any) -> int:
    if query.ndim not in (3, 4):
        raise FusionMetricError(f"query rank must be 3 or 4, got {query.ndim}")
    batch = 1 if query.ndim == 3 else query.shape[0]
    qlen, heads, qdim = query.shape[-3:]
    kv_heads, topk = indices.shape[-2:]
    if kv_heads == 0 or heads % kv_heads:
        raise FusionMetricError("query heads must be divisible by key/value heads")
    return 2 * batch * qlen * heads * topk * (qdim + int(value_dim))


def _expected_nnz(value: TensorMeta, sparsity: Any) -> float:
    return value.numel() * (1.0 - float(sparsity)) / 2.0


PRIMITIVES: dict[str, Callable[..., Any]] = {
    "dim": _dim,
    "numel": _numel,
    "nbytes": _nbytes,
    "shape_numel": _shape_numel,
    "tensor_sum": _tensor_sum,
    "attention_pair_count": _attention_pair_count,
    "varlen_attention_flops": _varlen_attention_flops,
    "flash_mla_flops": _flash_mla_flops,
    "sparse_mla_flops": _sparse_mla_flops,
    "expected_nnz": _expected_nnz,
    "output_nbytes_like": lambda value, count: int(count) * value.element_size,
}
BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}
CMP_OPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Is: lambda a, b: a is b,
}


def _eval(node: ast.AST, values: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, values)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.List):
        return [_eval(x, values) for x in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval(x, values) for x in node.elts)
    if isinstance(node, ast.BinOp):
        return BIN_OPS[type(node.op)](
            _eval(node.left, values), _eval(node.right, values)
        )
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand, values)
        return +value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BoolOp):
        vals = [_eval(x, values) for x in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, values)
        for op, right_node in zip(node.ops, node.comparators):
            right = _eval(right_node, values)
            if not CMP_OPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        branch = node.body if _eval(node.test, values) else node.orelse
        return _eval(branch, values)
    if isinstance(node, ast.Call):
        return PRIMITIVES[node.func.id](*[_eval(x, values) for x in node.args])
    raise FusionMetricError(f"unsupported expression node: {type(node).__name__}")


def dtype_bytes(dtype: Any) -> int:
    sizes = {
        "float64": 8,
        "complex64": 8,
        "float32": 4,
        "bfloat16": 2,
        "float16": 2,
        "int64": 8,
        "int32": 4,
        "int16": 2,
        "bool": 1,
    }
    return sizes.get(canonical_dtype(dtype), 4)


VECTOR_INPUTS = {"window_size"}
INT_INPUTS = {
    "prompt_mask": 1,
    "output_mask": 1,
    "target": 8,
    "slot_mapping": 8,
    "topk_indices": 8,
    "token_expert_indices": 8,
    "expert_ids": 4,
    "sorted_ids": 4,
    "key_position": 4,
    "tokens_post_pad": 4,
    "cumulative_query_lengths": 4,
    "used_key_lengths": 4,
    "cache_sequence_lengths": 4,
}


def _shape_args(shape_detail: Any) -> tuple[list[Any], dict[str, Any]]:
    value = normalize(shape_detail)
    if isinstance(value, dict):
        return [], value
    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], dict):
        return list(value[0]), value[1]
    return list(value) if isinstance(value, list) else [value], {}


def _as_input(raw: Any, name: str, dtype: Any) -> Any:
    if name in VECTOR_INPUTS:
        return raw
    if raw is None or isinstance(raw, (bool, int, float, str)):
        return raw
    if isinstance(raw, list) and all(isinstance(x, (int, float)) for x in raw):
        size = INT_INPUTS.get(name, dtype_bytes(dtype))
        return TensorMeta(tuple(int(x) for x in raw), size)
    raise FusionMetricError(f"cannot infer tensor shape for input {name}: {raw!r}")


def calculate(
    op_name: str,
    dtype: Any,
    shape_detail: Any,
    formulas: Mapping[str, Formula],
) -> tuple[str, float, str]:
    name = canonical_op(op_name)
    if name not in formulas:
        raise FusionMetricError(f"no formula for operator {op_name}")
    formula = formulas[name]
    if not formula.enabled:
        raise FusionMetricError(f"formula disabled for {name}")
    args, kwargs = _shape_args(shape_detail)
    values: dict[str, Any] = {}
    for input_name, binding in formula.inputs.items():
        keyword = binding.get("keyword")
        raw = kwargs.get(keyword) if keyword else None
        position = binding.get("position")
        if raw is None and position is not None and position < len(args):
            raw = args[position]
        if raw is None and "default" in binding:
            raw = binding["default"]
        if (
            raw is None
            and "default" not in binding
            and input_name not in ("weight", "scale", "use_input_stats")
        ):
            raise FusionMetricError(f"missing input {input_name}")
        values[input_name] = _as_input(raw, input_name, dtype)
    # The workload's random non-zero count is intentionally replaced by its
    # fixed theoretical value.
    if name == "rwkv_mm_sparsity":
        values["key"] = values["key"]
    try:
        result = _eval(formula.tree, values)
    except FusionMetricError:
        raise
    except Exception as exc:
        raise FusionMetricError(
            f"formula {name} failed: {type(exc).__name__}: {exc}"
        ) from exc
    if isinstance(result, bool) or not isinstance(result, (int, float)) or result < 0:
        raise FusionMetricError(f"invalid result {result!r}")
    return formula.metric, float(result), formula.category


def _json_lines(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"\{.*\}", line)
        if not match:
            continue
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "op_name" in obj and "result" in obj:
            yield obj


def log_files(logs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in logs:
        path = path.expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("records/*.log")))
            files.extend(sorted(path.glob("raw/*.log")))
        elif path.is_file():
            files.append(path)
    return list(dict.fromkeys(files))


def read_rows(
    logs: Iterable[Path],
    ops: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in log_files(logs):
        for record in _json_lines(path):
            op = canonical_op(record["op_name"])
            if ops and op not in ops:
                continue
            for item in record.get("result", []):
                row = dict(item)
                row.update(
                    {
                        "op_name": op,
                        "dtype": canonical_dtype(record.get("dtype", "")),
                        "mode": record.get("mode", "N/A"),
                        "level": record.get("level", "N/A"),
                        "record_path": str(path),
                    }
                )
                rows.append(row)
    return rows


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temp = Path(stream.name)
    temp.replace(path)


def build_map(
    logs: list[Path],
    output: Path,
    update: Path | None,
    ops: set[str] | None,
) -> dict[str, Any]:
    formulas = load_formulas()
    record_files = log_files(logs)
    if not record_files:
        raise FusionMetricError(
            "no record logs found; inspect runner records/ and stdout/"
        )
    entries: dict[str, Any] = {}
    if update and update.exists():
        entries.update(
            json.loads(update.read_text(encoding="utf-8")).get("entries", {})
        )
    seen: dict[str, float] = {}
    rows = read_rows(record_files, ops)
    if not rows:
        raise FusionMetricError(
            "record logs contain no benchmark records; inspect stdout/"
        )
    for row in rows:
        if row.get("error_msg"):
            continue
        key = make_key(row["op_name"], row["dtype"], row.get("shape_detail"))
        metric, value, category = calculate(
            row["op_name"], row["dtype"], row.get("shape_detail"), formulas
        )
        if key in seen and seen[key] != value:
            raise FusionMetricError(f"conflicting value for key {key}")
        seen[key] = value
        entries[key] = {
            "category": category,
            "metric": metric,
            "value": int(value) if float(value).is_integer() else value,
        }
    result = {"generated_at": _timestamp(), "entries": entries}
    _atomic_json(output, result)
    return result


def _num(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _targets(pipeline: str, dtype: str, target_tflops: float | None) -> float:
    if target_tflops is not None:
        return target_tflops
    try:
        return TARGET_TFLOPS[(pipeline, dtype)]
    except KeyError as exc:
        raise FusionMetricError(
            f"no target TFLOPS for {pipeline}/{dtype}; pass --target-tflops"
        ) from exc


def _pipeline_for_category(category: str, override: str | None) -> str:
    if override is not None:
        return override
    try:
        return DEFAULT_PIPELINE_BY_CATEGORY[category]
    except KeyError as exc:
        raise FusionMetricError(f"unknown metric category: {category}") from exc


def _report_rows(
    rows: list[dict[str, Any]],
    mapping: Mapping[str, Any],
    pipeline: str | None,
    target_tflops: float | None,
    target_gbps: float,
    runtime: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    compute: list[dict[str, Any]] = []
    memory: list[dict[str, Any]] = []
    for row in rows:
        if row.get("error_msg"):
            continue
        key = make_key(row["op_name"], row["dtype"], row.get("shape_detail"))
        entry = mapping.get(key)
        if not entry:
            raise FusionMetricError(f"mapping key missing: {key}")
        gems = _num(row.get("latency"))
        torch_latency = _num(row.get("latency_base"))
        if (
            gems is None
            or torch_latency is None
            or gems <= 0
            or torch_latency <= 0
        ):
            raise FusionMetricError(f"invalid latency for {key}")
        value = float(entry["value"])
        shape = json.dumps(
            row.get("shape_detail"), ensure_ascii=False, separators=(",", ":")
        )
        facts = (runtime or {}).get(row["op_name"], {})
        row_pipeline = _pipeline_for_category(entry["category"], pipeline)
        common = {
            "pipeline": row_pipeline,
            "OMP线程数": facts.get("omp_threads", "N/A"),
            "warmup": facts.get("warmup", "N/A"),
            "iter": facts.get("iter", "N/A"),
        }
        if entry["category"] == "compute":
            target = _targets(row_pipeline, row["dtype"], target_tflops)
            expected = value / target / 1e9 if target else None
            out = {
                "算子名称": row["op_name"],
                "shape": shape,
                "dtype": row["dtype"],
                "Gems延迟(ms)": gems,
                "Torch延迟(ms)": torch_latency,
                "目标算力下期望延迟(ms)": expected,
                "目标达成率(%)": expected / gems * 100 if expected else None,
                "逻辑计算量(GFLOPs)": value / 1e9,
                "实测算力(TFLOPS)": value / gems / 1e9,
                "目标算力(TFLOPS)": target,
                "实测/期望延迟倍率": gems / expected if expected else None,
                "Torch Speedup": torch_latency / gems,
                "源码文件": SOURCE_FILES.get(row["op_name"], "N/A"),
                **common,
                "benchmark模式": row.get("mode", "N/A"),
                "备注": "",
            }
            compute.append(out)
        else:
            expected = value / target_gbps / 1e6 if target_gbps else None
            memory.append(
                {
                    "算子名称": row["op_name"],
                    "shape": shape,
                    "dtype": row["dtype"],
                    "Gems延迟(ms)": gems,
                    "Torch延迟(ms)": torch_latency,
                    "目标带宽下期望延迟(ms)": expected,
                    "目标达成率(%)": expected / gems * 100 if expected else None,
                    "逻辑数据量(GB)": value / 1e9,
                    "逻辑带宽(GB/s)": value / gems / 1e6,
                    "目标带宽(GB/s)": target_gbps,
                    "Torch逻辑带宽(GB/s)": value / torch_latency / 1e6,
                    "Speedup": torch_latency / gems,
                    "源码文件": SOURCE_FILES.get(row["op_name"], "N/A"),
                    **common,
                    "benchmark模式": row.get("mode", "N/A"),
                    "备注": "",
                }
            )
    return compute, memory


def _write_csv(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _runtime_facts(logs: list[Path]) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    for candidate in logs:
        root = candidate if candidate.is_dir() else candidate.parent
        manifest = root / "run.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        common = {
            key: data.get(key, "N/A")
            for key in ("omp_threads", "warmup", "iter")
        }
        for item in data.get("operators", []):
            facts[item.get("op_name")] = common
    return facts


def _coverage_rows(
    rows: list[dict[str, Any]],
    mapping: Mapping[str, Any],
    pipeline: str | None,
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        key = make_key(row["op_name"], row["dtype"], row.get("shape_detail"))
        entry = mapping.get(key)
        category = entry.get("category", "N/A") if entry else "N/A"
        row_pipeline = (
            _pipeline_for_category(category, pipeline) if entry else "N/A"
        )
        result.append(
            {
                "算子名称": row["op_name"],
                "类别": category,
                "状态": "READY" if entry else "MAP_MISSING",
                "原因": "" if entry else "mapping key missing",
                "测试文件": "N/A",
                "marker": "N/A",
                "pipeline": row_pipeline,
                "NUMA节点": "N/A",
                "stdout": "N/A",
                "record log": row.get("record_path", "N/A"),
            }
        )
    return result


def _publish_snapshot(output_dir: Path, snapshot: Path, names: Iterable[str]) -> None:
    """Atomically publish copies while retaining the immutable snapshot."""
    for name in names:
        temporary = output_dir / f".{name}.tmp"
        shutil.copyfile(snapshot / name, temporary)
        temporary.replace(output_dir / name)


def _default_output_dir(logs: list[Path]) -> Path:
    """Place a report under the supplied run directory by default."""
    if len(logs) == 1:
        source = logs[0].expanduser().resolve()
        if source.is_dir():
            run_dir = source
        elif source.parent.name in {"records", "raw", "stdout"}:
            run_dir = source.parent.parent
        else:
            run_dir = source.parent
        return run_dir / "report"
    return Path.cwd() / "fusion_metrics_report"


def report(
    logs: list[Path],
    map_path: Path,
    output_dir: Path,
    pipeline: str | None,
    target_tflops: float | None,
    target_gbps: float,
    ops: set[str] | None = None,
) -> Path:
    map_data = json.loads(map_path.read_text(encoding="utf-8"))
    mapping = map_data.get("entries", {})
    record_files = log_files(logs)
    if not record_files:
        raise FusionMetricError(
            "no record logs found; inspect runner records/ and stdout/"
        )
    rows = read_rows(record_files, ops)
    if not rows:
        raise FusionMetricError(
            "record logs contain no benchmark records; inspect stdout/"
        )
    compute, memory = _report_rows(
        rows,
        mapping,
        pipeline,
        target_tflops,
        target_gbps,
        _runtime_facts(logs),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "runs" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    snapshot.mkdir(parents=True, exist_ok=True)
    _write_csv(
        snapshot / "fused_compute_tflops.csv",
        COMPUTE_HEADER,
        compute,
    )
    _write_csv(
        snapshot / "fused_non_compute_bandwidth.csv",
        MEMORY_HEADER,
        memory,
    )
    _write_csv(
        snapshot / "coverage.csv",
        COVERAGE_HEADER,
        _coverage_rows(rows, mapping, pipeline),
    )
    _publish_snapshot(
        output_dir,
        snapshot,
        (
            "fused_compute_tflops.csv",
            "fused_non_compute_bandwidth.csv",
            "coverage.csv",
        ),
    )
    manifest = {
        "generated_at": _timestamp(),
        "pipeline": pipeline or "auto",
        "map_generated_at": map_data.get("generated_at"),
        "source_logs": [str(x) for x in logs],
        "rows": {"compute": len(compute), "memory": len(memory)},
    }
    _atomic_json(snapshot / "run.json", manifest)
    _atomic_json(output_dir / "run.json", manifest)
    return output_dir


def update_report(
    logs: list[Path],
    ops: set[str],
    map_path: Path,
    output_dir: Path,
    pipeline: str | None,
    target_tflops: float | None,
    target_gbps: float,
) -> Path:
    # Generate new selected rows first; only publish after all map/log checks pass.
    map_data = json.loads(map_path.read_text(encoding="utf-8"))
    mapping = map_data.get("entries", {})
    record_files = log_files(logs)
    if not record_files:
        raise FusionMetricError(
            "no record logs found; inspect runner records/ and stdout/"
        )
    new_rows = read_rows(record_files, ops)
    if not new_rows:
        raise FusionMetricError(
            "record logs contain no benchmark records; inspect stdout/"
        )
    found = {row["op_name"] for row in new_rows}
    missing_ops = {canonical_op(op) for op in ops} - found
    if missing_ops:
        names = ", ".join(sorted(missing_ops))
        raise FusionMetricError(f"selected operators missing from logs: {names}")
    seen_keys = set()
    for row in new_rows:
        key = make_key(row["op_name"], row["dtype"], row.get("shape_detail"))
        if key in seen_keys:
            raise FusionMetricError(f"duplicate workload key in update logs: {key}")
        seen_keys.add(key)

    new_compute, new_memory = _report_rows(
        new_rows,
        mapping,
        pipeline,
        target_tflops,
        target_gbps,
        _runtime_facts(logs),
    )

    def read_existing(name: str) -> list[dict[str, Any]]:
        path = output_dir / name
        if not path.exists():
            return []
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    selected = {canonical_op(op) for op in ops}
    compute = [
        row
        for row in read_existing("fused_compute_tflops.csv")
        if canonical_op(row.get("算子名称", "")) not in selected
    ] + new_compute
    memory = [
        row
        for row in read_existing("fused_non_compute_bandwidth.csv")
        if canonical_op(row.get("算子名称", "")) not in selected
    ] + new_memory
    coverage = [
        row
        for row in read_existing("coverage.csv")
        if canonical_op(row.get("算子名称", "")) not in selected
    ] + _coverage_rows(new_rows, mapping, pipeline)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "runs" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    snapshot.mkdir(parents=True, exist_ok=True)
    _write_csv(
        snapshot / "fused_compute_tflops.csv",
        COMPUTE_HEADER,
        compute,
    )
    _write_csv(
        snapshot / "fused_non_compute_bandwidth.csv",
        MEMORY_HEADER,
        memory,
    )
    _write_csv(snapshot / "coverage.csv", COVERAGE_HEADER, coverage)
    manifest = {
        "generated_at": _timestamp(),
        "pipeline": pipeline or "auto",
        "map_generated_at": map_data.get("generated_at"),
        "source_logs": [str(x) for x in logs],
        "updated_ops": sorted(selected),
        "rows": {"compute": len(new_compute), "memory": len(new_memory)},
    }
    _atomic_json(snapshot / "run.json", manifest)
    _publish_snapshot(
        output_dir,
        snapshot,
        (
            "fused_compute_tflops.csv",
            "fused_non_compute_bandwidth.csv",
            "coverage.csv",
        ),
    )
    _atomic_json(output_dir / "run.json", manifest)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-map")
    build.add_argument("--logs", nargs="+", type=Path, required=True)
    build.add_argument("--output-map", type=Path, default=MAP_PATH)
    build.add_argument("--update-map", type=Path)
    build.add_argument("--ops", nargs="+")
    for name in ("report", "update-report"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--logs", nargs="+", type=Path, required=True)
        cmd.add_argument(
            "--map",
            type=Path,
            default=MAP_PATH,
            help=f"Mapping file (default: {MAP_PATH})",
        )
        cmd.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="Report directory (default: <run-dir>/report)",
        )
        cmd.add_argument(
            "--pipeline",
            choices=["SME", "SVE"],
            default=None,
            help="Override the automatic compute=SME, memory=SVE mapping",
        )
        cmd.add_argument("--target-tflops", type=float)
        cmd.add_argument("--target-gbps", type=float, default=30.72)
        if name == "update-report":
            cmd.add_argument("--ops", nargs="+", required=True)
    args = parser.parse_args(argv)
    if args.command in {"report", "update-report"}:
        args.output_dir = args.output_dir or _default_output_dir(args.logs)
    try:
        if args.command == "build-map":
            build_map(
                args.logs,
                args.output_map,
                args.update_map,
                set(args.ops) if args.ops else None,
            )
        elif args.command == "report":
            report(
                args.logs,
                args.map,
                args.output_dir,
                args.pipeline,
                args.target_tflops,
                args.target_gbps,
            )
        else:
            update_report(
                args.logs,
                set(args.ops),
                args.map,
                args.output_dir,
                args.pipeline,
                args.target_tflops,
                args.target_gbps,
            )
    except (OSError, KeyError, ValueError, yaml.YAMLError, FusionMetricError) as exc:
        parser.error(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
