from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FormulaInfo:
    formula_id: str
    source: str
    formula: str
    description: str
    ops: tuple[str, ...]


@dataclass(frozen=True)
class TflopsInfo:
    tflops: float | None
    flops: int | float | None
    source: str
    formula_id: str | None


UNARY_POINTWISE_OPS = {
    "abs",
    "absolute",
    "acos",
    "angle",
    "arcsinh",
    "arctanh",
    "asinh",
    "atan",
    "bitwise_not",
    "ceil",
    "celu",
    "cos",
    "cosh",
    "digamma",
    "dropout",
    "elu",
    "erf",
    "exp",
    "exp2",
    "expm1",
    "floor",
    "gelu",
    "hardsigmoid",
    "hardswish",
    "i0",
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "leaky_relu",
    "log",
    "log_sigmoid",
    "log1p",
    "log10",
    "logical_not",
    "neg",
    "reciprocal",
    "relu",
    "relu6",
    "round",
    "rsqrt",
    "selu",
    "sgn",
    "sigmoid",
    "signbit",
    "silu",
    "sin",
    "sinh",
    "softplus",
    "softshrink",
    "special_i0e",
    "special_i1",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "threshold",
}

BINARY_POINTWISE_OPS = {
    "add",
    "allclose",
    "bitwise_and",
    "bitwise_or",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "div",
    "dunder_ior",
    "dunder_or",
    "eq",
    "equal",
    "floor_divide",
    "fmin",
    "gcd",
    "ge",
    "greater",
    "gt",
    "hypot",
    "isclose",
    "le",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logaddexp",
    "lt",
    "maximum",
    "minimum",
    "mul",
    "ne",
    "polar",
    "pow",
    "remainder",
    "sub",
}

SCALAR_BINARY_POINTWISE_OPS = {
    "pow_scalar",
}

REDUCTION_OP_FACTORS = {
    "sum": 1,
    "prod": 1,
    "max": 1,
    "min": 1,
    "amax": 1,
    "amin": 1,
    "all": 1,
    "any": 1,
    "argmax": 1,
    "argmin": 1,
    "mean": 2,
    "var": 3,
    "std": 4,
    "softmax": 4,
    "log_softmax": 4,
    "scaled_softmax": 5,
    "safe_softmax": 4,
}

NORMALIZATION_FACTORS = {
    "batch_norm": 5,
    "batch_norm_backward": 8,
    "fused_add_rms_norm": 6,
    "group_norm": 5,
    "instance_norm": 5,
    "layer_norm": 5,
    "rms_norm": 4,
    "skip_layer_norm": 6,
    "vector_norm": 2,
}

FUSED_FACTORS = {
    "addcdiv": 3,
    "addcmul": 3,
    "gelu_and_mul": 6,
    "silu_and_mul": 4,
    "where": 1,
}

ATTENTION_OPS = {
    "flash_attention_forward",
    "flash_mla",
    "scaled_dot_product_attention",
}


FORMULAS: dict[str, FormulaInfo] = {
    "benchmark:mm": FormulaInfo(
        "benchmark:mm",
        "benchmark",
        "2 * M * N * K",
        "Existing FlagGems BlasBenchmark formula for matrix multiply.",
        ("mm",),
    ),
    "benchmark:bmm": FormulaInfo(
        "benchmark:bmm",
        "benchmark",
        "2 * B * M * N * K",
        "Existing FlagGems BlasBenchmark formula for batched matrix multiply.",
        ("bmm",),
    ),
    "benchmark:addmm": FormulaInfo(
        "benchmark:addmm",
        "benchmark",
        "M * N * (2 * K + 1)",
        "Existing FlagGems BlasBenchmark formula for addmm.",
        ("addmm",),
    ),
    "benchmark:baddbmm": FormulaInfo(
        "benchmark:baddbmm",
        "benchmark",
        "B * M * N * (2 * K + 1)",
        "Existing FlagGems BaddbmmBenchmark formula.",
        ("baddbmm",),
    ),
    "benchmark:grouped_mm": FormulaInfo(
        "benchmark:grouped_mm",
        "benchmark",
        "sum_g(2 * M_g * N_g * K_g)",
        "Existing grouped matmul formula.",
        ("grouped_mm",),
    ),
    "benchmark:unary_pointwise": FormulaInfo(
        "benchmark:unary_pointwise",
        "benchmark",
        "numel",
        "Existing UnaryPointwiseBenchmark formula.",
        tuple(sorted(UNARY_POINTWISE_OPS)),
    ),
    "benchmark:binary_pointwise": FormulaInfo(
        "benchmark:binary_pointwise",
        "benchmark",
        "2 * numel",
        "Existing BinaryPointwiseBenchmark formula.",
        tuple(sorted(BINARY_POINTWISE_OPS)),
    ),
    "benchmark:scalar_binary_pointwise": FormulaInfo(
        "benchmark:scalar_binary_pointwise",
        "benchmark",
        "numel",
        "Existing ScalarBinaryPointwiseBenchmark formula.",
        tuple(sorted(SCALAR_BINARY_POINTWISE_OPS)),
    ),
    "benchmark:tex_glu_forward": FormulaInfo(
        "benchmark:tex_glu_forward",
        "benchmark",
        "input_numel",
        "Existing TexGluForwardBenchmark formula.",
        ("geglu", "glu", "reglu", "swiglu"),
    ),
    "benchmark:tex_glu_backward": FormulaInfo(
        "benchmark:tex_glu_backward",
        "benchmark",
        "2 * input_numel",
        "Existing TexGluBackwardBenchmark formula.",
        ("dgeglu", "dglu", "dreglu", "dswiglu"),
    ),
    "derived:blas_like": FormulaInfo(
        "derived:blas_like",
        "derived",
        "op-specific BLAS-like formula",
        "Reference formulas for BLAS-like ops not covered by current benchmark output.",
        ("addmv", "addr", "dot", "kron", "mv", "outer", "vdot"),
    ),
    "derived:conv": FormulaInfo(
        "derived:conv",
        "derived",
        "2 * output_elements * (Cin / groups) * kernel_elements",
        "Reference convolution FLOPs.",
        ("conv1d", "conv2d", "conv3d", "conv_depthwise2d"),
    ),
    "derived:pooling": FormulaInfo(
        "derived:pooling",
        "derived",
        "output_elements * kernel_elements",
        "Reference pooling work estimate.",
        ("avg_pool2d", "max_pool2d_with_indices"),
    ),
    "derived:reduction": FormulaInfo(
        "derived:reduction",
        "derived",
        "op_factor * input_numel",
        "Reference reduction and softmax formulas.",
        tuple(sorted(REDUCTION_OP_FACTORS)),
    ),
    "derived:normalization": FormulaInfo(
        "derived:normalization",
        "derived",
        "op_factor * input_numel",
        "Reference normalization formulas.",
        tuple(sorted(NORMALIZATION_FACTORS)),
    ),
    "derived:fused_pointwise": FormulaInfo(
        "derived:fused_pointwise",
        "derived",
        "op_factor * numel",
        "Reference fused pointwise formulas.",
        tuple(sorted(FUSED_FACTORS)),
    ),
    "derived:attention": FormulaInfo(
        "derived:attention",
        "derived",
        "4 * batch_heads * q_len * kv_len * head_dim",
        "Reference attention formula using QK^T and PV matmul main terms.",
        tuple(sorted(ATTENTION_OPS)),
    ),
}

OP_FORMULA_IDS = {
    op: formula.formula_id
    for formula in FORMULAS.values()
    for op in formula.ops
}


def resolve_tflops(
    op_name: Any,
    shape_detail: Any,
    latency_ms: Any,
    benchmark_tflops: Any,
) -> TflopsInfo:
    op = str(op_name or "")
    formula_id = OP_FORMULA_IDS.get(op)
    if _is_number(benchmark_tflops):
        tflops = float(benchmark_tflops)
        return TflopsInfo(
            tflops=tflops,
            flops=_flops_from_tflops(tflops, latency_ms),
            source="benchmark",
            formula_id=formula_id,
        )

    flops = estimate_flops(op, shape_detail)
    if flops is None:
        return TflopsInfo(
            tflops=None,
            flops=None,
            source="none",
            formula_id=None,
        )

    latency = _float_or_none(latency_ms)
    tflops = None
    if latency is not None and latency > 0:
        tflops = float(flops) / latency / 1e12 * 1e3
    return TflopsInfo(
        tflops=tflops,
        flops=flops,
        source="derived",
        formula_id=formula_id,
    )


def estimate_flops(op_name: str, shape_detail: Any) -> int | float | None:
    op = str(op_name or "")
    if op in {"mm", "bmm", "addmm", "baddbmm"}:
        return _matmul_flops(op, shape_detail)
    if op in {"addmv", "addr", "dot", "kron", "mv", "outer", "vdot"}:
        return _blas_like_flops(op, shape_detail)
    if op in {"conv1d", "conv2d", "conv3d", "conv_depthwise2d"}:
        return _conv_flops(shape_detail)
    if op in {"avg_pool2d", "max_pool2d_with_indices"}:
        return _pooling_flops(shape_detail)
    if op in ATTENTION_OPS:
        return _attention_flops(shape_detail)
    if op in FUSED_FACTORS:
        return _first_numel(shape_detail, FUSED_FACTORS[op])
    if op in REDUCTION_OP_FACTORS:
        return _first_numel(shape_detail, REDUCTION_OP_FACTORS[op])
    if op in NORMALIZATION_FACTORS:
        return _first_numel(shape_detail, NORMALIZATION_FACTORS[op])
    if op in UNARY_POINTWISE_OPS:
        return _first_numel(shape_detail, 1)
    if op in BINARY_POINTWISE_OPS:
        return _first_numel(shape_detail, 2)
    return None


def formula_catalog() -> list[dict[str, Any]]:
    return [asdict(formula) for formula in FORMULAS.values()]


def _matmul_flops(op: str, shape_detail: Any) -> int | None:
    shapes = _tensor_shapes(shape_detail)
    if op == "mm" and len(shapes) >= 2 and len(shapes[0]) == 2 and len(shapes[1]) == 2:
        m, k = shapes[0]
        _, n = shapes[1]
        return 2 * m * n * k
    if op == "bmm" and len(shapes) >= 2 and len(shapes[0]) == 3 and len(shapes[1]) == 3:
        b, m, k = shapes[0]
        _, _, n = shapes[1]
        return 2 * b * m * n * k
    if op == "addmm" and len(shapes) >= 3 and len(shapes[1]) == 2 and len(shapes[2]) == 2:
        m, k = shapes[1]
        _, n = shapes[2]
        return m * n * (2 * k + 1)
    if op == "baddbmm" and len(shapes) >= 3 and len(shapes[1]) == 3 and len(shapes[2]) == 3:
        b, m, k = shapes[1]
        _, _, n = shapes[2]
        return b * m * n * (2 * k + 1)
    return None


def _blas_like_flops(op: str, shape_detail: Any) -> int | None:
    shapes = _tensor_shapes(shape_detail)
    if op in {"dot", "vdot"} and shapes:
        numel = _prod(shapes[0])
        return 2 * numel
    if op == "outer" and len(shapes) >= 2:
        return _prod(shapes[0]) * _prod(shapes[1])
    if op == "mv" and len(shapes) >= 2 and len(shapes[0]) == 2:
        m, n = shapes[0]
        return 2 * m * n
    if op == "addmv" and len(shapes) >= 3 and len(shapes[1]) == 2:
        m, n = shapes[1]
        return m * (2 * n + 1)
    if op == "addr" and len(shapes) >= 3:
        return 2 * _prod(shapes[0])
    if op == "kron" and len(shapes) >= 2:
        return _prod(shapes[0]) * _prod(shapes[1])
    return None


def _conv_flops(shape_detail: Any) -> int | None:
    kwargs = _first_dict(shape_detail)
    input_shape = _shape_from_key(kwargs, "input")
    weight_shape = _shape_from_key(kwargs, "weight")
    if input_shape is None or weight_shape is None or len(input_shape) < 3:
        shapes = _tensor_shapes(shape_detail)
        if len(shapes) < 2:
            return None
        input_shape, weight_shape = shapes[0], shapes[1]

    spatial_rank = len(input_shape) - 2
    if spatial_rank <= 0 or len(weight_shape) != spatial_rank + 2:
        return None

    groups = int(kwargs.get("groups", 1)) if isinstance(kwargs, dict) else 1
    stride = _expand_param(kwargs.get("stride", 1), spatial_rank)
    padding = _expand_padding(kwargs.get("padding", 0), spatial_rank)
    dilation = _expand_param(kwargs.get("dilation", 1), spatial_rank)
    if stride is None or padding is None or dilation is None:
        return None

    batch = input_shape[0]
    out_channels = weight_shape[0]
    kernel_shape = weight_shape[2:]
    out_spatial = []
    for input_size, kernel, step, pad, dil in zip(
        input_shape[2:], kernel_shape, stride, padding, dilation
    ):
        out_spatial.append(_conv_output_size(input_size, kernel, step, pad, dil))
    return 2 * batch * out_channels * _prod(out_spatial) * (weight_shape[1] * _prod(kernel_shape))


def _pooling_flops(shape_detail: Any) -> int | None:
    shapes = _tensor_shapes(shape_detail)
    if not shapes or len(shapes[0]) < 4:
        return None
    kwargs = _first_dict(shape_detail)
    input_shape = shapes[0]
    spatial_rank = len(input_shape) - 2
    kernel = _expand_param(kwargs.get("kernel_size", 1), spatial_rank)
    stride = _expand_param(kwargs.get("stride", kernel), spatial_rank)
    padding = _expand_param(kwargs.get("padding", 0), spatial_rank)
    dilation = _expand_param(kwargs.get("dilation", 1), spatial_rank)
    if kernel is None or stride is None or padding is None or dilation is None:
        return None
    out_spatial = [
        _conv_output_size(size, kern, step, pad, dil)
        for size, kern, step, pad, dil in zip(
            input_shape[2:], kernel, stride, padding, dilation
        )
    ]
    return input_shape[0] * input_shape[1] * _prod(out_spatial) * _prod(kernel)


def _attention_flops(shape_detail: Any) -> int | None:
    shapes = _tensor_shapes(shape_detail)
    if len(shapes) < 3:
        return None
    q_shape, k_shape = shapes[0], shapes[1]
    if len(q_shape) < 3 or len(k_shape) < 3:
        return None
    head_dim = q_shape[-1]
    q_len = q_shape[-2]
    kv_len = k_shape[-2]
    batch_heads = _prod(q_shape[:-2])
    return 4 * batch_heads * q_len * kv_len * head_dim


def _first_numel(shape_detail: Any, factor: int | float) -> int | float | None:
    shapes = _tensor_shapes(shape_detail)
    if not shapes:
        return None
    return factor * _prod(shapes[0])


def _tensor_shapes(value: Any) -> list[list[int]]:
    shapes: list[list[int]] = []

    def visit(item: Any) -> None:
        if _is_shape(item):
            shapes.append([int(dim) for dim in item])
            return
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return shapes


def _first_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_dict(item)
            if found:
                return found
    return {}


def _shape_from_key(mapping: dict[str, Any], key: str) -> list[int] | None:
    value = mapping.get(key)
    if _is_shape(value):
        return [int(dim) for dim in value]
    return None


def _is_shape(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(_is_int_like(dim) for dim in value)
    )


def _expand_param(value: Any, rank: int) -> list[int] | None:
    if value is None:
        return [1] * rank
    if isinstance(value, str):
        return None
    if _is_int_like(value):
        return [int(value)] * rank
    if isinstance(value, (list, tuple)) and len(value) == rank and all(
        _is_int_like(item) for item in value
    ):
        return [int(item) for item in value]
    return None


def _expand_padding(value: Any, rank: int) -> list[int] | None:
    if value == "valid":
        return [0] * rank
    if value == "same":
        return [0] * rank
    return _expand_param(value, rank)


def _conv_output_size(
    input_size: int, kernel: int, stride: int, padding: int, dilation: int
) -> int:
    return math.floor((input_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


def _prod(values: list[int] | tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _flops_from_tflops(tflops: float, latency_ms: Any) -> float | None:
    latency = _float_or_none(latency_ms)
    if latency is None:
        return None
    return tflops * latency / 1e3 * 1e12


def _float_or_none(value: Any) -> float | None:
    if not _is_number(value):
        return None
    return float(value)


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_int_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return False
    return parsed == value


def main() -> int:
    parser = argparse.ArgumentParser(description="Print FlagGems TFLOPS formulas.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the formula catalog as JSON.",
    )
    args = parser.parse_args()
    catalog = formula_catalog()
    if args.json:
        print(json.dumps(catalog, indent=2, sort_keys=True))
    else:
        for item in catalog:
            ops = ", ".join(item["ops"])
            print(f"{item['formula_id']}: {item['formula']} [{ops}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
