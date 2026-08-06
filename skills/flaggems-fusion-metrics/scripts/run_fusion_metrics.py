#!/usr/bin/env python3
"""Run and report FlagGems fusion metrics.

Usage:
  python3 run_fusion_metrics.py --ops silu_and_mul
  python3 run_fusion_metrics.py --all
  python3 run_fusion_metrics.py --export-formulas /tmp/formulas.md

Environment overrides: AGENT_DIR, TRITON_REPO_DIR.
Optional defaults: scripts/.env (CLI overrides .env).
Design: ../references/design.md
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SKILL_DIR = Path(__file__).resolve().parents[1]
AGENTFORTRITONCPU_DIR = SKILL_DIR.parents[1]
AGENT_DIR = Path(
    os.environ.get("AGENT_DIR", str(AGENTFORTRITONCPU_DIR.parent))
).expanduser().resolve()
REPO_ROOT = Path(
    os.environ.get("TRITON_REPO_DIR", str(AGENT_DIR / "triton-cpu"))
).expanduser().resolve()
FLAGGEMS_ROOT = REPO_ROOT / "FlagGems"
BENCHMARK_DIR = FLAGGEMS_ROOT / "benchmark"
SOURCE_ROOT = FLAGGEMS_ROOT / "src" / "flag_gems"
FORMULA_CONFIG_PATH = BENCHMARK_DIR / "fusion_metrics_formulas.yaml"
ENV_SCRIPT = (
    AGENTFORTRITONCPU_DIR
    / "skills"
    / "environment"
    / "scripts"
    / "triton-cpu-env.sh"
)
DEFAULT_OUTPUT_DIR = AGENT_DIR / "logs" / "fusion_metrics"
DEFAULT_SHAPE_FILE = BENCHMARK_DIR / "core_shapes.yaml"
DOTENV_PATH = Path(__file__).resolve().with_name(".env")

DOTENV_KEYS = {
    "FUSION_WARMUP": ("warmup", int),
    "FUSION_ITER": ("iterations", int),
    "FUSION_MODE": ("mode", str),
    "FUSION_LEVEL": ("level", str),
    "FUSION_DTYPES": ("dtypes", str),
    "FUSION_JOBS": ("jobs", int),
    "FUSION_OMP_THREADS": ("omp_threads", int),
    "FUSION_NUMA_NODES": ("numa_nodes", str),
    "FUSION_TARGET_TFLOPS": ("target_tflops", float),
    "FUSION_TARGET_GBPS": ("target_gbps", float),
    "FUSION_TIMEOUT": ("timeout", int),
    "FUSION_SHAPE_FILE": ("shape_file", Path),
    "FUSION_OUTPUT_DIR": ("output_dir", Path),
}

if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from fusion_metrics import FormulaConfig, FusionFormulaError, load_formula_config

COMPUTE_HEADER = [
    "算子名称",
    "shape",
    "dtype",
    "Gems延迟(ms)",
    "Torch延迟(ms)",
    "目标算力下期望延迟(ms)",
    "目标达成率(%)",
    "逻辑计算量(GFLOPs)",
    "实测算力(TFLOPS)",
    "目标算力(TFLOPS)",
    "实测/期望延迟倍率",
    "Torch Speedup",
    "源码文件",
    "pipeline",
    "OMP线程数",
    "warmup",
    "iter",
    "benchmark模式",
    "备注",
]

MEMORY_HEADER = [
    "算子名称",
    "shape",
    "dtype",
    "Gems延迟(ms)",
    "Torch延迟(ms)",
    "目标带宽下期望延迟(ms)",
    "目标达成率(%)",
    "逻辑数据量(GB)",
    "逻辑带宽(GB/s)",
    "目标带宽(GB/s)",
    "Torch逻辑带宽(GB/s)",
    "Speedup",
    "源码文件",
    "pipeline",
    "OMP线程数",
    "warmup",
    "iter",
    "benchmark模式",
    "备注",
]

COVERAGE_HEADER = [
    "算子名称",
    "类别",
    "状态",
    "原因",
    "测试文件",
    "marker",
    "pipeline",
    "NUMA节点",
    "stdout",
    "record log",
]

# Guide default is FP32.  These benchmarks expose only BF16 (or use BF16 as
# their stable reference dtype) in the current tree.
DEFAULT_DTYPE_FALLBACKS = {
    "flash_attention_forward": "bfloat16",
    "flash_attn_varlen_func": "bfloat16",
    "flash_mla": "bfloat16",
    "flash_mla_sparse_fwd": "bfloat16",
    "sparse_mla_fwd_interface": "bfloat16",
    "get_scheduler_metadata": "bfloat16",
}

# Scaled compute ceilings used by the report when no explicit target is
# provided. Keep this keyed by the pipeline/dtype pair selected per spec.
TARGET_TFLOPS_SCALE = 0.2 / 304 * 32
THEORETICAL_TARGET_TFLOPS = {
    ("SME", "float32"): 120.0 * TARGET_TFLOPS_SCALE,
    ("SME", "bfloat16"): 240.0 * TARGET_TFLOPS_SCALE,
    ("SVE", "float32"): 34.0 * TARGET_TFLOPS_SCALE,
}

SOURCE_FILES = {
    "flash_attention_forward": "ops/attention.py",
    "flash_attn_varlen_func": "ops/attention.py",
    "flash_mla": "fused/flash_mla.py",
    "sparse_mla_fwd_interface": "fused/DSA/sparse_mla.py",
    "apply_repetition_penalties": "fused/apply_repetition_penalties.py",
    "apply_rotary_pos_emb": "fused/rotary_embedding.py",
    "concat_and_cache_mla": "fused/concat_and_cache_mla.py",
    "cross_entropy_loss": "fused/cross_entropy_loss.py",
    "dgeglu": "fused/geglu.py",
    "dreglu": "fused/reglu.py",
    "fused_add_rms_norm": "fused/fused_add_rms_norm.py",
    "geglu": "fused/geglu.py",
    "gelu_and_mul": "fused/gelu_and_mul.py",
    "get_scheduler_metadata": "ops/get_scheduler_metadata.py",
    "instance_norm": "fused/instance_norm.py",
    "moe_align_block_size_triton": "fused/moe_align_block_size.py",
    "moe_sum": "fused/moe_sum.py",
    "reglu": "fused/reglu.py",
    "reshape_and_cache": "fused/reshape_and_cache.py",
    "reshape_and_cache_flash": "fused/reshape_and_cache_flash.py",
    "rwkv_ka_fusion": "fused/rwkv_ka_fusion.py",
    "rwkv_mm_sparsity": "fused/rwkv_mm_sparsity.py",
    "silu_and_mul": "fused/silu_and_mul.py",
    "silu_and_mul_out": "fused/silu_and_mul.py",
    "skip_layer_norm": "fused/skip_layernorm.py",
    "topk_softmax": "fused/topk_softmax.py",
    "weight_norm": "fused/weight_norm.py",
}

META_MARKERS = {
    "parametrize",
    "skip",
    "skipif",
    "xfail",
    "usefixtures",
    "filterwarnings",
    "timeout",
    "tryfirst",
    "trylast",
}

PRINT_LOCK = threading.Lock()
FORMULA_CONFIG: FormulaConfig | None = None


# YAML names stay bound to benchmark argument positions.  These names only
# change the formula text shown for the public operator signature.
DISPLAY_INPUT_NAMES = {
    "apply_repetition_penalties": {"penalties": "repetition_penalties"},
    "apply_rotary_pos_emb": {
        "query": "q",
        "key": "k",
        "cosine": "cos",
        "sine": "sin",
    },
    "concat_and_cache_mla": {
        "kv_content": "kv_c",
        "key_position": "k_pe",
        "cache_dtype": "kv_cache_dtype",
    },
    "cross_entropy_loss": {"logits": "inp"},
    "dgeglu": {"gradient": "grad_output", "input": "input_tensor"},
    "dreglu": {"gradient": "grad_output", "input": "input_tensor"},
    "fused_add_rms_norm": {"input": "x"},
    "flash_attn_varlen_func": {
        "query": "q",
        "cumulative_query_lengths": "cu_seqlens_q",
        "used_key_lengths": "seqused_k",
    },
    "flash_mla": {
        "query_length": "s_q",
        "cache_sequence_lengths": "cache_seqlens",
        "query_heads": "h_q",
        "query_key_dim": "d",
        "value_dim": "dv",
    },
    "flash_mla_sparse_fwd": {"query": "q", "value_dim": "d_v"},
    "geglu": {"input": "input_tensor"},
    "gelu_and_mul": {"input": "x", "other": "y"},
    "get_scheduler_metadata": {"cache_sequence_lengths": "seqused_k"},
    "instance_norm": {"running_variance": "running_var"},
    "moe_align_block_size_triton": {
        "sorted_ids": "sorted_token_ids",
        "tokens_post_pad": "num_tokens_post_pad",
    },
    "reglu": {"input": "input_tensor"},
    "rwkv_ka_fusion": {
        "key": "k",
        "key_key": "kk",
        "attention": "a",
        "key_attention": "ka",
    },
    "rwkv_mm_sparsity": {"key": "k", "value": "v"},
    "silu_and_mul": {"input": "A", "other": "B"},
    "silu_and_mul_out": {"input": "A", "other": "B"},
    "skip_layer_norm": {"input": "x"},
    "sparse_mla_fwd_interface": {"query": "q", "value_dim": "d_v"},
    "weight_norm": {"value": "v", "scale": "g"},
}


@dataclass(frozen=True)
class TestSpec:
    op_name: str
    category: str
    marker: str | None
    test_file: str | None
    pipeline: str
    dtype: str | None
    dtype_fallback: bool = False


@dataclass(frozen=True)
class NumaBinding:
    node: int
    cpus: tuple[int, ...]


@dataclass
class ExecutionResult:
    spec: TestSpec
    status: str
    returncode: int | None
    timed_out: bool
    note: str | None
    numa_node: int | None
    cpu_list: str | None
    stdout_path: str | None
    record_path: str | None
    command: list[str]
    environment: dict[str, str]
    records: list[dict[str, Any]]


def _print(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def _formula_config() -> FormulaConfig:
    global FORMULA_CONFIG
    if FORMULA_CONFIG is None:
        FORMULA_CONFIG = load_formula_config(FORMULA_CONFIG_PATH)
    return FORMULA_CONFIG


def _csv_notes() -> list[str]:
    config = _formula_config()
    return [
        "说明：公式位于 FlagGems/benchmark/fusion_metrics_formulas.yaml。",
        f"说明：本次配置 SHA-256：{config.sha256}。",
    ]


def _marker_from_decorator(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        node = node.func
    if not isinstance(node, ast.Attribute) or node.attr in META_MARKERS:
        return None
    parent = node.value
    if not isinstance(parent, ast.Attribute) or parent.attr != "mark":
        return None
    if not isinstance(parent.value, ast.Name) or parent.value.id != "pytest":
        return None
    return node.attr


def discover_markers() -> dict[str, str]:
    discovered: dict[str, str] = {}
    for path in sorted(BENCHMARK_DIR.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                marker = _marker_from_decorator(decorator)
                if marker and marker not in discovered:
                    discovered[marker] = path.name
    return discovered


def _parse_cpu_list(value: str) -> list[int]:
    cpus: list[int] = []
    for item in value.strip().split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(item))
    return sorted(set(cpus))


def _format_cpu_list(cpus: Iterable[int]) -> str:
    ordered = sorted(set(cpus))
    if not ordered:
        return ""
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def discover_numa_topology() -> dict[int, tuple[int, ...]]:
    topology: dict[int, tuple[int, ...]] = {}
    sysfs_root = Path("/sys/devices/system/node")
    for node_dir in sorted(sysfs_root.glob("node[0-9]*")):
        match = re.fullmatch(r"node(\d+)", node_dir.name)
        if not match:
            continue
        cpulist = node_dir / "cpulist"
        if cpulist.exists():
            cpus = tuple(_parse_cpu_list(cpulist.read_text(encoding="utf-8")))
            if cpus:
                topology[int(match.group(1))] = cpus
    if topology:
        return topology

    if shutil.which("numactl") is None:
        return topology
    result = subprocess.run(
        ["numactl", "-H"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    for line in result.stdout.splitlines():
        match = re.match(r"\s*node\s+(\d+)\s+cpus:\s*(.*)$", line)
        if match:
            cpus = tuple(_parse_cpu_list(match.group(2)))
            if cpus:
                topology[int(match.group(1))] = cpus
    return topology


def make_bindings(
    jobs: int,
    omp_threads: int,
    requested_nodes: str | None,
    topology: dict[int, tuple[int, ...]] | None = None,
) -> list[NumaBinding]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if omp_threads < 1:
        raise ValueError("omp-threads must be positive")
    topology = topology if topology is not None else discover_numa_topology()
    if requested_nodes:
        node_ids = [int(item.strip()) for item in requested_nodes.split(",") if item.strip()]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("numa-nodes must not contain duplicates")
    else:
        node_ids = sorted(topology)
    if len(node_ids) < jobs:
        raise ValueError(
            f"need {jobs} distinct NUMA nodes, available: {sorted(topology)}"
        )

    bindings = []
    for node in node_ids[:jobs]:
        if node not in topology:
            raise ValueError(f"NUMA node {node} is unavailable")
        cpus = topology[node]
        if len(cpus) < omp_threads:
            raise ValueError(
                f"NUMA node {node} has {len(cpus)} CPUs, need {omp_threads}"
            )
        bindings.append(NumaBinding(node=node, cpus=tuple(cpus[:omp_threads])))

    cpu_sets = [set(binding.cpus) for binding in bindings]
    for index, left in enumerate(cpu_sets):
        for right in cpu_sets[index + 1 :]:
            if left & right:
                raise ValueError("NUMA CPU bindings overlap")
    return bindings


def _dtype_for_op(op_name: str, explicit: list[str] | None) -> tuple[str, bool]:
    if explicit:
        if len(explicit) != 1:
            raise ValueError(
                "fusion metrics requires one fixed dtype per run; "
                "rerun separate runs for multiple dtypes"
            )
        return explicit[0], False
    fallback = DEFAULT_DTYPE_FALLBACKS.get(op_name)
    return fallback or "float32", fallback is not None


def _dtype_attempts(spec: TestSpec, args: argparse.Namespace) -> list[TestSpec]:
    """Return the explicit-dtype retry chain for one operator."""
    if args.dtypes is None:
        return [spec]

    attempts = [spec]
    if _canonical_dtype(spec.dtype) != "bfloat16":
        attempts.append(replace(spec, dtype="bfloat16", dtype_fallback=True))
    attempts.append(replace(spec, dtype=None, dtype_fallback=True))
    return attempts


def _dtype_fallback_message(
    op_name: str,
    failed_dtype: str | None,
    next_dtype: str | None,
) -> str:
    failed = failed_dtype or "bfloat16"
    if next_dtype == "bfloat16":
        return f"[fusion] {op_name}: 不支持 {failed}，回退到 bfloat16 类型执行"
    return f"[fusion] {op_name}: 不支持 {failed}，取消类型指定执行"


def build_specs(
    selected_ops: Iterable[str] | None,
    explicit_dtypes: list[str] | None,
) -> tuple[list[TestSpec], list[dict[str, str]]]:
    config = _formula_config()
    all_ops = sorted(config.formulas)
    requested_names = (
        all_ops if selected_ops is None else list(dict.fromkeys(selected_ops))
    )
    discovered = discover_markers()
    specs: list[TestSpec] = []
    unavailable: list[dict[str, str]] = []
    seen: set[str] = set()
    for requested_name in requested_names:
        try:
            formula = config.resolve(requested_name)
        except FusionFormulaError:
            unavailable.append(
                {"op_name": requested_name, "reason": "不在 fusion operator 列表中"}
            )
            continue
        op_name = formula.name
        if op_name in seen:
            continue
        seen.add(op_name)
        category = formula.category
        pipeline = "SME" if category == "compute" else "SVE"
        if not formula.enabled:
            unavailable.append(
                {"op_name": op_name, "reason": formula.exclude_reason or "disabled"}
            )
            continue
        marker = next(
            (
                candidate
                for candidate in (op_name, *formula.aliases)
                if candidate in discovered
            ),
            None,
        )
        test_file = discovered.get(marker) if marker else None
        if test_file is None:
            unavailable.append(
                {
                    "op_name": op_name,
                    "reason": (
                        "未发现 marker "
                        + "/".join((op_name, *formula.aliases))
                    ),
                }
            )
            continue
        dtype, fallback = _dtype_for_op(op_name, explicit_dtypes)
        specs.append(
            TestSpec(
                op_name=op_name,
                category=category,
                marker=marker,
                test_file=test_file,
                pipeline=pipeline,
                dtype=dtype,
                dtype_fallback=fallback,
            )
        )
    return specs, unavailable


def _pytest_args(spec: TestSpec, args: argparse.Namespace) -> list[str]:
    metrics = ["latency_base", "latency", "speedup"]
    metrics.append("tflops" if spec.category == "compute" else "gbps")
    try:
        shape_file_arg = str(args.shape_file.relative_to(BENCHMARK_DIR))
    except ValueError:
        shape_file_arg = str(args.shape_file)
    values = [
        spec.test_file or "",
        "-q",
        "--tb=no",
        "-m",
        spec.marker or "",
        "--mode",
        args.mode,
        "--level",
        args.level,
        "--warmup",
        str(args.warmup),
        "--iter",
        str(args.iterations),
        "--shape_file",
        shape_file_arg,
        "--record",
        "log",
    ]
    if spec.dtype:
        values.extend(["--dtypes", spec.dtype])
    for metric in metrics:
        values.extend(["--metrics", metric])
    return values


def _expected_record_path(pytest_args: list[str]) -> Path:
    cmd_args = [
        arg.replace(".py", "").replace("=", "_").replace("/", "_")
        for arg in pytest_args
    ]
    return BENCHMARK_DIR / ("result_{}.log".format("_".join(cmd_args))).replace(
        "_-", "-"
    )


def build_command(
    spec: TestSpec,
    args: argparse.Namespace,
    binding: NumaBinding,
) -> tuple[list[str], list[str]]:
    pytest_args = _pytest_args(spec, args)
    if shutil.which("numactl") is None or shutil.which("taskset") is None:
        raise RuntimeError("numactl and taskset are required for NUMA-isolated runs")
    prefix = [
        "numactl",
        f"--cpunodebind={binding.node}",
        f"--membind={binding.node}",
        "taskset",
        "-c",
        _format_cpu_list(binding.cpus),
    ]
    command = [
        *prefix,
        getattr(args, "python_executable", sys.executable),
        "-m",
        "pytest",
        *pytest_args,
    ]
    return command, pytest_args


def _record_payloads(path: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if not path.exists():
        return payloads
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("[INFO] "):
            continue
        body = line[len("[INFO] ") :].strip()
        if not body.startswith("{"):
            continue
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "op_name" in data and "result" in data:
            payloads.append(data)
    return payloads


def _failure_note(output: str, returncode: int | None) -> str:
    prefixes = (
        "ModuleNotFoundError:",
        "ImportError:",
        "RuntimeError:",
        "ValueError:",
        "AssertionError:",
        "Failed:",
        "ERROR",
    )
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefixes):
            return stripped[:500]
    for line in reversed(output.splitlines()):
        if line.strip():
            return f"pytest exit code {returncode}: {line.strip()[:420]}"
    return f"pytest exit code {returncode}"


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _load_runtime_environment() -> tuple[dict[str, str], str]:
    """Load the environment used by direct benchmark invocations."""
    if not ENV_SCRIPT.is_file():
        raise RuntimeError(f"环境脚本不存在: {ENV_SCRIPT}")

    bootstrap_env = os.environ.copy()
    bootstrap_env.setdefault("AGENT_DIR", str(REPO_ROOT.parent))
    bootstrap_env.setdefault("TRITON_REPO_DIR", str(REPO_ROOT))
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" >/dev/null 2>&1 && env -0',
            "fusion-metrics-env",
            str(ENV_SCRIPT),
        ],
        env=bootstrap_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        output = result.stdout.decode(errors="replace").strip()
        detail = f": {output[-800:]}" if output else ""
        raise RuntimeError(f"加载环境脚本失败 {ENV_SCRIPT}{detail}")

    loaded_env: dict[str, str] = {}
    for item in result.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        loaded_env[key.decode()] = value.decode(errors="surrogateescape")
    python_executable = Path(loaded_env.get("VENV_DIR", "")) / "bin" / "python3"
    if not python_executable.is_file():
        python_executable = Path(sys.executable)
    if not python_executable.is_file():
        raise RuntimeError(f"Python 可执行文件不存在: {python_executable}")
    return loaded_env, str(python_executable)


def _environment_snapshot(env: dict[str, str]) -> dict[str, str]:
    keys = (
        "AGENT_DIR",
        "VENV_DIR",
        "PATH",
        "PYTHONPATH",
        "LLVM_INSTALL_DIR",
        "LLVM_BINARY_DIR",
        "TRITON_REPO_DIR",
        "TRITON_PLUGIN_DIRS",
        "TRITON_SHARED_OPT_PATH",
        "TRITON_USE_SHARED_BACKEND",
        "TRITON_DISABLE_LINE_INFO",
        "GEMS_VENDOR",
        "OMP_NUM_THREADS",
        "TRITON_SHARED_FORCE_SME_PIPELINE",
        "TRITON_SHARED_FORCE_SVE_PIPELINE",
        "FLAGGEMS_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "FLAGGEMS_FUSION_METRICS_CONFIG",
        "FLAGGEMS_FUSION_METRICS_KEY",
    )
    return {key: env[key] for key in keys if key in env}


def _child_env(
    spec: TestSpec,
    run_dir: Path,
    args: argparse.Namespace,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base_env if base_env is not None else os.environ)
    env["OMP_NUM_THREADS"] = str(args.omp_threads)
    env.pop("TRITON_SHARED_FORCE_SME_PIPELINE", None)
    env.pop("TRITON_SHARED_FORCE_SVE_PIPELINE", None)
    env[f"TRITON_SHARED_FORCE_{spec.pipeline}_PIPELINE"] = "1"
    cache_root = run_dir / "cache" / spec.op_name
    env["FLAGGEMS_CACHE_DIR"] = str(cache_root / "flaggems")
    env["TRITON_CACHE_DIR"] = str(cache_root / "triton")
    env["FLAGGEMS_FUSION_METRICS_CONFIG"] = str(_formula_config().path)
    env["FLAGGEMS_FUSION_METRICS_KEY"] = spec.op_name
    python_path = [
        str(REPO_ROOT / "python"),
        str(FLAGGEMS_ROOT),
        str(FLAGGEMS_ROOT / "src"),
        str(REPO_ROOT),
    ]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    return env


def run_one(
    spec: TestSpec,
    binding: NumaBinding,
    args: argparse.Namespace,
    run_dir: Path,
    attempt: int = 0,
) -> ExecutionResult:
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    command, pytest_args = build_command(spec, args, binding)
    expected_log = _expected_record_path(pytest_args)
    expected_log.unlink(missing_ok=True)
    child_env = _child_env(
        spec, run_dir, args, getattr(args, "runtime_environment", None)
    )
    _print(
        f"[fusion] start {spec.op_name} pipeline={spec.pipeline} "
        f"dtype={spec.dtype or 'auto'} numa={binding.node} "
        f"cpus={_format_cpu_list(binding.cpus)}"
    )
    output = ""
    returncode: int | None = None
    timed_out = False
    process = subprocess.Popen(
        command,
        cwd=BENCHMARK_DIR,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        try:
            output, _ = process.communicate(timeout=args.timeout)
            returncode = process.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            partial = exc.output or ""
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            output = partial
            _terminate_process(process)
            tail, _ = process.communicate()
            output += tail or ""
            returncode = process.returncode
    finally:
        if process.stdout is not None:
            process.stdout.close()

    stdout_path = raw_dir / f"{spec.op_name}.attempt{attempt}.stdout"
    stdout_path.write_text(output, encoding="utf-8")
    record_path: Path | None = None
    if expected_log.exists():
        record_path = raw_dir / f"{spec.op_name}.attempt{attempt}.log"
        shutil.move(str(expected_log), str(record_path))
    records = _record_payloads(record_path) if record_path else []

    if timed_out:
        status = "timeout"
        note = f"pytest 超时，限制 {args.timeout}s"
    elif returncode == 0 and records:
        status = "passed"
        note = None
    elif returncode != 0 and records:
        status = "partial"
        note = _failure_note(output, returncode)
    elif returncode == 0 and "skipped" in output.lower():
        status = "skipped"
        note = "pytest skipped"
    elif returncode != 0:
        status = "failed"
        note = _failure_note(output, returncode)
    else:
        status = "no-data"
        note = "没有生成 benchmark JSON record"
    if record_path is None:
        note = "; ".join(item for item in [note, "record log 不存在"] if item)
    _print(f"[fusion] done {spec.op_name}: {status}")
    return ExecutionResult(
        spec=spec,
        status=status,
        returncode=returncode,
        timed_out=timed_out,
        note=note,
        numa_node=binding.node,
        cpu_list=_format_cpu_list(binding.cpus),
        stdout_path=str(stdout_path.relative_to(run_dir)),
        record_path=str(record_path.relative_to(run_dir)) if record_path else None,
        command=command,
        environment=_environment_snapshot(child_env),
        records=records,
    )


def run_with_dtype_fallback(
    spec: TestSpec,
    binding: NumaBinding,
    args: argparse.Namespace,
    run_dir: Path,
) -> ExecutionResult:
    attempts = _dtype_attempts(spec, args)
    fallback_messages: list[str] = []
    result: ExecutionResult | None = None
    for attempt, attempt_spec in enumerate(attempts):
        result = run_one(attempt_spec, binding, args, run_dir, attempt)
        if result.status == "passed":
            break
        if attempt + 1 < len(attempts):
            message = _dtype_fallback_message(
                spec.op_name,
                attempt_spec.dtype,
                attempts[attempt + 1].dtype,
            )
            fallback_messages.append(message.removeprefix("[fusion] "))
            _print(message)

    assert result is not None
    if fallback_messages:
        result.note = "；".join(fallback_messages + ([result.note] if result.note else []))
    return result


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number(value: Any, digits: int = 6) -> str:
    number = _finite(value)
    return "N/A" if number is None else f"{number:.{digits}f}"


def _dtype_name(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value)
    return text.rsplit(".", 1)[-1]


def _canonical_dtype(value: Any) -> str:
    aliases = {
        "fp32": "float32",
        "bf16": "bfloat16",
    }
    name = _dtype_name(value).lower()
    return aliases.get(name, name)


def target_tflops_for_spec(
    spec: TestSpec,
    dtype: Any,
    override: float | None,
) -> float:
    if override is not None:
        return override
    key = (spec.pipeline, _canonical_dtype(dtype))
    try:
        return THEORETICAL_TARGET_TFLOPS[key]
    except KeyError as exc:
        supported = ", ".join(
            f"{pipeline}/{dtype}={value:g}"
            for (pipeline, dtype), value in sorted(THEORETICAL_TARGET_TFLOPS.items())
        )
        raise ValueError(
            f"no theoretical target TFLOPS for pipeline={spec.pipeline}, "
            f"dtype={_dtype_name(dtype)}; supported combinations: {supported}; "
            "set --target-tflops to override"
        ) from exc


def _shape_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _source_file(op_name: str, test_file: str | None) -> str:
    source = SOURCE_FILES.get(op_name)
    if source and (SOURCE_ROOT / source).exists():
        return source
    return f"benchmark/{test_file}" if test_file else "N/A"


def _row_note(spec: TestSpec) -> str:
    notes = [f"公式来源=fusion_metrics_formulas.yaml:{spec.op_name}"]
    if spec.dtype_fallback:
        notes.insert(0, f"dtype fallback，使用{spec.dtype or 'benchmark默认dtype'}")
    return "；".join(notes)


def build_metric_row(
    spec: TestSpec,
    metric: dict[str, Any],
    args: argparse.Namespace,
) -> list[str] | None:
    latency = _finite(metric.get("latency"))
    latency_base = _finite(metric.get("latency_base"))
    if latency is None or latency <= 0:
        return None
    common_tail = [
        _source_file(spec.op_name, spec.test_file),
        spec.pipeline,
        str(args.omp_threads),
        str(args.warmup),
        str(args.iterations),
        args.mode,
        _row_note(spec),
    ]
    shape = _shape_text(metric.get("shape_detail"))
    dtype = _dtype_name(metric.get("dtype", spec.dtype))
    if spec.category == "compute":
        logical_flops = _finite(metric.get("logical_flops"))
        if logical_flops is None:
            return None
        target_tflops = target_tflops_for_spec(
            spec, dtype, args.target_tflops
        )
        expected = logical_flops / (target_tflops * 1e9)
        achievement = expected / latency * 100
        measured = logical_flops / (latency * 1e9)
        ratio = latency / expected if expected > 0 else None
        speedup = latency_base / latency if latency_base else None
        return [
            spec.op_name,
            shape,
            dtype,
            _number(latency),
            _number(latency_base),
            _number(expected),
            _number(achievement),
            _number(logical_flops / 1e9),
            _number(measured),
            _number(target_tflops),
            _number(ratio),
            _number(speedup),
            *common_tail,
        ]

    logical_bytes = _finite(metric.get("logical_bytes"))
    if logical_bytes is None:
        return None
    expected = logical_bytes / (args.target_gbps * 1e6)
    achievement = expected / latency * 100
    bandwidth = logical_bytes / (latency * 1e6)
    torch_bandwidth = (
        logical_bytes / (latency_base * 1e6) if latency_base and latency_base > 0 else None
    )
    speedup = latency_base / latency if latency_base else None
    return [
        spec.op_name,
        shape,
        dtype,
        _number(latency),
        _number(latency_base),
        _number(expected),
        _number(achievement),
        _number(logical_bytes / 1e9),
        _number(bandwidth),
        _number(args.target_gbps),
        _number(torch_bandwidth),
        _number(speedup),
        *common_tail,
    ]


def build_rows(
    executions: Iterable[ExecutionResult],
    args: argparse.Namespace,
) -> tuple[list[list[str]], list[list[str]], dict[str, int]]:
    compute_rows: list[list[str]] = []
    memory_rows: list[list[str]] = []
    missing_metric_data: dict[str, int] = {}
    for execution in executions:
        if execution.status != "passed":
            continue
        for payload in execution.records:
            payload_dtype = payload.get("dtype", execution.spec.dtype)
            for metric in payload.get("result", []):
                if metric.get("error_msg"):
                    continue
                metric = dict(metric)
                metric["dtype"] = payload_dtype
                row = build_metric_row(execution.spec, metric, args)
                if row is None:
                    missing_metric_data[execution.spec.op_name] = (
                        missing_metric_data.get(execution.spec.op_name, 0) + 1
                    )
                    continue
                if execution.spec.category == "compute":
                    compute_rows.append(row)
                else:
                    memory_rows.append(row)
    return compute_rows, memory_rows, missing_metric_data


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
        handle.write("\n")
        for note in _csv_notes():
            handle.write(note + "\n")
    temporary.replace(path)


def write_coverage(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows([COVERAGE_HEADER, *rows])
    temporary.replace(path)


def read_data_rows(path: Path, header: list[str]) -> list[list[str]]:
    if not path.exists():
        return []
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row == header:
                continue
            if len(row) == len(header) and row and row[0] != "说明：":
                rows.append(row)
    return rows


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _execution_manifest(execution: ExecutionResult) -> dict[str, Any]:
    data = asdict(execution)
    data["spec"] = asdict(execution.spec)
    data.pop("records", None)
    return data


def _config_signature(args: argparse.Namespace) -> dict[str, Any]:
    config = _formula_config()
    return {
        "mode": args.mode,
        "level": args.level,
        "warmup": args.warmup,
        "iter": args.iterations,
        "dtypes": args.dtypes,
        "shape_file": str(args.shape_file.resolve()),
        "omp_threads": args.omp_threads,
        "target_tflops": args.target_tflops,
        "theoretical_target_tflops": {
            f"{pipeline}/{dtype}": value
            for (pipeline, dtype), value in sorted(
                THEORETICAL_TARGET_TFLOPS.items()
            )
        },
        "target_gbps": args.target_gbps,
        "jobs": args.jobs,
        "numa_nodes": args.numa_nodes,
        "timeout": args.timeout,
        "output_dir": str(args.output_dir),
        "formula_sha256": config.sha256,
    }


def _manifest(
    args: argparse.Namespace,
    requested_ops: list[str],
    specs: list[TestSpec],
    unavailable: list[dict[str, str]],
    executions: list[ExecutionResult],
    run_dir: Path,
    compute_rows: list[list[str]],
    memory_rows: list[list[str]],
    missing_metric_data: dict[str, int],
) -> dict[str, Any]:
    config = _formula_config()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_ops": requested_ops,
        "config_signature": _config_signature(args),
        "formula_config": {
            "path": str(config.path),
            "sha256": config.sha256,
        },
        "specs": [asdict(spec) for spec in specs],
        "unavailable": unavailable,
        "executions": [_execution_manifest(execution) for execution in executions],
        "missing_metric_data": missing_metric_data,
        "row_counts": {
            "compute": len(compute_rows),
            "memory": len(memory_rows),
        },
        "run_dir": str(run_dir),
        "files": {
            "compute": "fused_compute_tflops.csv",
            "memory": "fused_non_compute_bandwidth.csv",
            "coverage": "coverage.csv",
        },
    }


def _coverage_rows(
    specs: list[TestSpec],
    unavailable: list[dict[str, str]],
    executions: list[ExecutionResult],
    run_dir: Path,
) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in unavailable:
        rows.append(
            [
                item["op_name"],
                "N/A",
                "SKIPPED",
                item["reason"],
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
            ]
        )
    for execution in executions:
        spec = execution.spec
        rows.append(
            [
                spec.op_name,
                spec.category,
                execution.status.upper(),
                execution.note or "",
                spec.test_file or "N/A",
                spec.marker or "N/A",
                spec.pipeline,
                str(execution.numa_node) if execution.numa_node is not None else "N/A",
                execution.stdout_path or "N/A",
                execution.record_path or "N/A",
            ]
        )
    known = {row[0] for row in rows}
    for spec in specs:
        if spec.op_name not in known:
            rows.append(
                [
                    spec.op_name,
                    spec.category,
                    "NOT_RUN",
                    "未执行",
                    spec.test_file or "N/A",
                    spec.marker or "N/A",
                    spec.pipeline,
                    "N/A",
                    "N/A",
                    "N/A",
                ]
            )
    return rows


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"


def _display_name(formula: Any, name: str) -> str:
    return DISPLAY_INPUT_NAMES.get(formula.name, {}).get(name, name)


def _pair_definition() -> str:
    return (
        "where P(Q, K, C, L, R) = Σᵢ₌₀⁽Q⁻¹⁾ max(0, Eᵢ - Sᵢ); "
        "Cᵢ = i + (K - Q if C or (L >= 0 and R >= 0) else 0); "
        "Sᵢ = 0 if L < 0 else max(0, Cᵢ - L); "
        "Eᵢ = min(K, Cᵢ + 1) if C else K; "
        "Eᵢ = min(Eᵢ, Cᵢ + R + 1) if R >= 0"
    )


def _format_pair_count(
    query_length: str,
    key_length: str,
    causal: str,
    window_left: str,
    window_right: str,
) -> str:
    return f"P({query_length}, {key_length}, {causal}, {window_left}, {window_right})"


class _FormulaFormatter:
    def __init__(self, formula: Any):
        self.formula = formula
        self.pair_used = False

    def render(self, expression: str) -> str:
        return self.visit(ast.parse(expression, mode="eval").body)

    def visit(self, node: ast.AST) -> str:
        method = getattr(self, f"visit_{type(node).__name__}", self.visit_generic)
        return method(node)

    def visit_Name(self, node: ast.Name) -> str:
        return _display_name(self.formula, node.id)

    def visit_Constant(self, node: ast.Constant) -> str:
        return repr(node.value)

    def visit_List(self, node: ast.List) -> str:
        return "[" + ", ".join(self.visit(item) for item in node.elts) + "]"

    def visit_Tuple(self, node: ast.Tuple) -> str:
        return "(" + ", ".join(self.visit(item) for item in node.elts) + ")"

    def visit_BinOp(self, node: ast.BinOp) -> str:
        op = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
        }[type(node.op)]
        return f"({self.visit(node.left)} {op} {self.visit(node.right)})"

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        op = "+" if isinstance(node.op, ast.UAdd) else "-"
        return f"({op}{self.visit(node.operand)})"

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        op = " and " if isinstance(node.op, ast.And) else " or "
        return "(" + op.join(self.visit(value) for value in node.values) + ")"

    def visit_Compare(self, node: ast.Compare) -> str:
        parts = [self.visit(node.left)]
        for op, right in zip(node.ops, node.comparators):
            parts.extend((self._compare_op(op), self.visit(right)))
        return "(" + " ".join(parts) + ")"

    @staticmethod
    def _compare_op(op: ast.cmpop) -> str:
        return {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Is: "is",
            ast.IsNot: "is not",
        }[type(op)]

    def visit_IfExp(self, node: ast.IfExp) -> str:
        return f"({self.visit(node.body)} if {self.visit(node.test)} else {self.visit(node.orelse)})"

    def visit_Subscript(self, node: ast.Subscript) -> str:
        return f"{self.visit(node.value)}[{self.visit(node.slice)}]"

    def visit_Slice(self, node: ast.Slice) -> str:
        return ":"

    def visit_Call(self, node: ast.Call) -> str:
        name = node.func.id if isinstance(node.func, ast.Name) else "call"
        args = [self.visit(arg) for arg in node.args]
        if name == "nbytes":
            return f"({args[0]}.numel() * {args[0]}.element_size())"
        if name == "numel":
            return f"{args[0]}.numel()"
        if name == "dim":
            return f"{args[0]}.shape[{args[1]}]"
        if name == "shape_numel":
            return f"product({args[0]})"
        if name == "tensor_sum":
            return f"sum({args[0]})"
        if name == "count_nonzero":
            return f"{args[0]}.count_nonzero()"
        if name == "output_nbytes_like":
            return f"({args[1]} * {args[0]}.element_size())"
        if name == "attention_pair_count":
            self.pair_used = True
            return _format_pair_count(*args)
        if name == "varlen_attention_flops":
            self.pair_used = True
            query, cumulative, used, causal, window = args
            pair = _format_pair_count(
                f"({cumulative}[b + 1] - {cumulative}[b])",
                f"{used}[b]",
                causal,
                f"{window}[0]",
                f"{window}[1]",
            )
            return (
                f"4 * {query}.shape[1] * {query}.shape[2] * "
                f"Σᵦ₌₀⁽ˡᵉⁿ({used})⁻¹⁾ {pair}"
            )
        if name == "sparse_mla_flops":
            query, indices, value_dim = args
            rank = f"len({query}.shape)"
            batch = f"(1 if {rank} == 3 else {query}.shape[0])"
            length = f"({query}.shape[0] if {rank} == 3 else {query}.shape[1])"
            heads = f"({query}.shape[1] if {rank} == 3 else {query}.shape[2])"
            return (
                f"2 * B * S * H * {indices}.shape[-1] * "
                f"({query}.shape[-1] + {value_dim}); "
                f"B = {batch}; S = {length}; H = {heads}"
            )
        return f"{name}({', '.join(args)})"

    def visit_generic(self, node: ast.AST) -> str:
        return ast.unparse(node)


def _format_formula(formula: Any) -> str:
    formatter = _FormulaFormatter(formula)
    expression = formatter.render(formula.expression)
    main, separator, definitions = expression.partition("; ")
    if formatter.pair_used:
        definitions = f"{definitions}; " if definitions else ""
        definitions += _pair_definition()
    try:
        main = ast.unparse(ast.parse(main, mode="eval").body)
    except SyntaxError:
        pass
    expression = main + (f"; {definitions}" if separator or definitions else "")
    return f"{formula.metric} = {expression}"


def export_formulas(config: FormulaConfig, output: Path) -> None:
    output = output.expanduser().resolve()
    lines = [
        "# FlagGems fusion metrics formulas",
        "",
        "| Product function | Category | Formula | Status |",
        "| --- | --- | --- | --- |",
    ]
    for formula in config.formulas.values():
        status = (
            "enabled"
            if formula.enabled
            else f"disabled: {formula.exclude_reason}"
        )
        cells = [
            formula.name,
            formula.category,
            f"`{_format_formula(formula)}`",
            status,
        ]
        lines.append(
            "| "
            + " | ".join(str(cell).replace("|", r"\|") for cell in cells)
            + " |"
        )
    _atomic_write_text(output, "\n".join(lines) + "\n")


class DotenvError(ValueError):
    pass


def _dotenv_error(path: Path, line: int, key: str, detail: str) -> DotenvError:
    return DotenvError(f"{path}:{line}: {key}: {detail}")


def _read_dotenv(path: Path) -> dict[str, tuple[str, int]]:
    if not path.exists():
        return {}
    values: dict[str, tuple[str, int]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise _dotenv_error(path, line_number, line, "expected KEY=VALUE")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key not in DOTENV_KEYS:
            raise _dotenv_error(path, line_number, key, "unknown key")
        if key in values:
            raise _dotenv_error(path, line_number, key, "duplicate key")
        raw_value = raw_value.strip()
        if raw_value[:1] in {"'", '"'}:
            quote = raw_value[0]
            closing = raw_value.find(quote, 1)
            if closing < 0 or raw_value[closing + 1 :].strip().lstrip("#").strip():
                raise _dotenv_error(path, line_number, key, "invalid quoted value")
            value = raw_value[1:closing]
        else:
            value = raw_value.split("#", 1)[0].rstrip()
        values[key] = (value, line_number)
    return values


def _absolute_dotenv_path(path: Path, line: int, key: str, value: str) -> Path:
    if "$" in value:
        raise _dotenv_error(path, line, key, "variable interpolation is not allowed")
    resolved = Path(value).expanduser()
    if not resolved.is_absolute():
        raise _dotenv_error(path, line, key, "path must be absolute or start with ~")
    return resolved.resolve()


def _dotenv_defaults(path: Path) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for key, (value, line) in _read_dotenv(path).items():
        if not value:
            continue
        dest, converter = DOTENV_KEYS[key]
        try:
            converted = converter(value)
        except ValueError as exc:
            raise _dotenv_error(path, line, key, f"invalid value {value!r}") from exc
        if converter is Path:
            converted = _absolute_dotenv_path(path, line, key, value)
        elif key == "FUSION_DTYPES":
            if any(character.isspace() for character in value) or "," in value:
                raise _dotenv_error(path, line, key, "exactly one dtype is allowed")
            converted = [value]
        elif key == "FUSION_NUMA_NODES":
            try:
                nodes = [int(item.strip()) for item in value.split(",") if item.strip()]
            except ValueError as exc:
                raise _dotenv_error(path, line, key, "invalid NUMA node list") from exc
            if (
                not nodes
                or any(node < 0 for node in nodes)
                or len(nodes) != len(set(nodes))
            ):
                raise _dotenv_error(path, line, key, "invalid NUMA node list")
        if key == "FUSION_MODE" and converted not in {"kernel", "operator", "wrapper"}:
            raise _dotenv_error(path, line, key, f"invalid value {value!r}")
        if key == "FUSION_LEVEL" and converted not in {"core", "comprehensive"}:
            raise _dotenv_error(path, line, key, f"invalid value {value!r}")
        if key == "FUSION_WARMUP" and converted < 0:
            raise _dotenv_error(path, line, key, "must be >= 0")
        if key in {"FUSION_ITER", "FUSION_JOBS", "FUSION_OMP_THREADS", "FUSION_TIMEOUT"}:
            if converted < 1:
                raise _dotenv_error(path, line, key, "must be positive")
        if key in {"FUSION_TARGET_TFLOPS", "FUSION_TARGET_GBPS"} and converted <= 0:
            raise _dotenv_error(path, line, key, "must be positive")
        defaults[dest] = converted
    return defaults


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"Optional defaults: {DOTENV_PATH}; CLI > .env > code defaults.",
    )
    parser.add_argument("--all", action="store_true", help="run all fusion operators")
    parser.add_argument("--ops", nargs="+", help="run selected fusion operators")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iter", dest="iterations", type=int, default=5)
    parser.add_argument("--mode", choices=["kernel", "operator", "wrapper"], default="kernel")
    parser.add_argument("--level", choices=["core", "comprehensive"], default="core")
    parser.add_argument("--dtypes", nargs="+", default=None)
    parser.add_argument("--shape-file", type=Path, default=DEFAULT_SHAPE_FILE)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--omp-threads", type=int, default=32)
    parser.add_argument("--numa-nodes", default=None, help="comma-separated NUMA node IDs")
    parser.add_argument(
        "--target-tflops",
        type=float,
        default=None,
        help="override the pipeline/dtype theoretical compute target",
    )
    # HBM 80, DDR5 * 4 = 4800 * 8 * 4 = 30.72 GB/s
    parser.add_argument("--target-gbps", type=float, default=30.72)
    parser.add_argument("--timeout", type=int, default=18000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--export-formulas",
        type=Path,
        metavar="OUTPUT.md",
        help="validate YAML and export formula Markdown without benchmark preflight",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="do not merge --ops into the full result",
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing")
    try:
        parser.set_defaults(**_dotenv_defaults(DOTENV_PATH))
    except (DotenvError, OSError) as exc:
        parser.error(str(exc))
    args = parser.parse_args(argv)
    if args.export_formulas and (args.all or args.ops or args.dry_run):
        parser.error("--export-formulas cannot be combined with run selection")
    if args.all and args.ops:
        parser.error("--all and --ops are mutually exclusive")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be >= 0 and iter must be positive")
    if args.jobs < 1 or args.omp_threads < 1 or args.timeout < 1:
        parser.error("jobs, omp-threads, and timeout must be positive")
    if args.target_tflops is not None and args.target_tflops <= 0:
        parser.error("target-tflops must be positive")
    if args.target_gbps <= 0:
        parser.error("target-gbps must be positive")
    for option, value in (
        ("shape-file", args.shape_file),
        ("output-dir", args.output_dir),
    ):
        expanded = value.expanduser()
        if not expanded.is_absolute():
            parser.error(f"{option} must be absolute or start with ~")
        setattr(args, option.replace("-", "_"), expanded.resolve())
    if not args.export_formulas and not args.shape_file.exists():
        parser.error(f"shape file does not exist: {args.shape_file}")
    return args


def _execute(
    specs: list[TestSpec],
    args: argparse.Namespace,
    run_dir: Path,
    bindings: list[NumaBinding],
) -> list[ExecutionResult]:
    if not specs:
        return []
    workers = min(args.jobs, len(specs))
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, spec in enumerate(specs):
            future = executor.submit(
                run_with_dtype_fallback,
                spec,
                bindings[index % len(bindings)],
                args,
                run_dir,
            )
            futures[future] = spec
        results = [future.result() for future in as_completed(futures)]
    return sorted(results, key=lambda item: item.spec.op_name)


def _publish(path: Path, source: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(path)


def _write_local_outputs(
    run_dir: Path,
    compute_rows: list[list[str]],
    memory_rows: list[list[str]],
    coverage_rows: list[list[str]],
) -> None:
    write_csv(run_dir / "fused_compute_tflops.csv", COMPUTE_HEADER, compute_rows)
    write_csv(run_dir / "fused_non_compute_bandwidth.csv", MEMORY_HEADER, memory_rows)
    write_coverage(run_dir / "coverage.csv", coverage_rows)


def _merge_single_run(
    output_dir: Path,
    manifest: dict[str, Any],
    selected_ops: set[str],
    compute_rows: list[list[str]],
    memory_rows: list[list[str]],
) -> None:
    base_manifest_path = output_dir / "run.json"
    if not base_manifest_path.exists():
        raise RuntimeError("全量结果不存在；先执行 --all，或使用 --no-update")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("config_signature") != manifest.get("config_signature"):
        raise RuntimeError("单项重跑配置与全量结果不一致，拒绝混合更新")

    compute_path = output_dir / "fused_compute_tflops.csv"
    memory_path = output_dir / "fused_non_compute_bandwidth.csv"
    if not compute_path.exists() or not memory_path.exists():
        raise RuntimeError("全量 CSV 不完整；先执行 --all，或使用 --no-update")
    old_compute = [
        row
        for row in read_data_rows(compute_path, COMPUTE_HEADER)
        if row[0] not in selected_ops
    ]
    old_memory = [
        row
        for row in read_data_rows(memory_path, MEMORY_HEADER)
        if row[0] not in selected_ops
    ]
    write_csv(compute_path, COMPUTE_HEADER, [*old_compute, *compute_rows])
    write_csv(memory_path, MEMORY_HEADER, [*old_memory, *memory_rows])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = _formula_config()
    except FusionFormulaError as exc:
        _print(f"[fusion] formula validation failed: {exc}")
        return 2
    if args.export_formulas:
        export_formulas(config, args.export_formulas)
        _print(f"[fusion] formulas exported: {args.export_formulas.expanduser().resolve()}")
        return 0
    try:
        args.runtime_environment, args.python_executable = _load_runtime_environment()
    except RuntimeError as exc:
        _print(f"[fusion] environment preflight failed: {exc}")
        return 2
    requested_ops = list(args.ops) if args.ops else None
    specs, unavailable = build_specs(requested_ops, args.dtypes)
    if args.ops:
        unknown = [
            item["op_name"]
            for item in unavailable
            if item["reason"] == "不在 fusion operator 列表中"
        ]
        if unknown:
            _print(f"[fusion] unknown fusion operator(s): {', '.join(unknown)}")
            return 2
    try:
        for spec in specs:
            if spec.category == "compute" and args.dtypes is None:
                target_tflops_for_spec(spec, spec.dtype, args.target_tflops)
    except ValueError as exc:
        _print(f"[fusion] target TFLOPS preflight failed: {exc}")
        return 2
    runtime_jobs = min(args.jobs, max(1, len(specs)))
    try:
        bindings = make_bindings(runtime_jobs, args.omp_threads, args.numa_nodes)
    except (RuntimeError, ValueError) as exc:
        _print(f"[fusion] NUMA preflight failed: {exc}")
        return 2

    if args.dry_run:
        _print(
            f"[fusion] operators={len(specs)} unavailable={len(unavailable)} jobs={runtime_jobs}"
        )
        for index, spec in enumerate(specs):
            binding = bindings[index % len(bindings)]
            command, _ = build_command(spec, args, binding)
            _print(" ".join(command))
        for item in unavailable:
            _print(f"[fusion] skipped {item['op_name']}: {item['reason']}")
        return 0

    run_dir = args.output_dir / "runs" / _run_id()
    run_dir.mkdir(parents=True, exist_ok=False)
    executions = _execute(specs, args, run_dir, bindings)
    compute_rows, memory_rows, missing_metric_data = build_rows(executions, args)
    coverage_rows = _coverage_rows(specs, unavailable, executions, run_dir)
    _write_local_outputs(run_dir, compute_rows, memory_rows, coverage_rows)

    manifest = _manifest(
        args,
        [spec.op_name for spec in specs] + [item["op_name"] for item in unavailable],
        specs,
        unavailable,
        executions,
        run_dir,
        compute_rows,
        memory_rows,
        missing_metric_data,
    )
    manifest_path = run_dir / "run.json"
    _atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))

    is_single = bool(args.ops)
    selected_ops = {spec.op_name for spec in specs}
    selected_failed = any(execution.status != "passed" for execution in executions)
    selected_without_rows = any(
        execution.status == "passed"
        and execution.spec.op_name not in {row[0] for row in [*compute_rows, *memory_rows]}
        for execution in executions
    )
    if not is_single:
        _publish(args.output_dir / "fused_compute_tflops.csv", run_dir / "fused_compute_tflops.csv")
        _publish(
            args.output_dir / "fused_non_compute_bandwidth.csv",
            run_dir / "fused_non_compute_bandwidth.csv",
        )
        _publish(args.output_dir / "coverage.csv", run_dir / "coverage.csv")
        _publish(args.output_dir / "run.json", manifest_path)
    elif not args.no_update and selected_ops and not selected_failed and not selected_without_rows:
        try:
            _merge_single_run(args.output_dir, manifest, selected_ops, compute_rows, memory_rows)
        except RuntimeError as exc:
            _print(f"[fusion] result merge rejected: {exc}")
            return 2
        manifest["merged_into"] = str(args.output_dir)
        manifest["updated_ops"] = sorted(selected_ops)
        _atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
        _publish(args.output_dir / "coverage.csv", run_dir / "coverage.csv")
        _publish(args.output_dir / "run.json", manifest_path)
    else:
        _print("[fusion] single-run result kept standalone; full CSV was not modified")

    _print(f"[fusion] run artifacts: {run_dir}")
    _print(
        f"[fusion] rows: compute={len(compute_rows)} memory={len(memory_rows)} "
        f"unavailable={len(unavailable)}"
    )
    return 0 if not selected_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
