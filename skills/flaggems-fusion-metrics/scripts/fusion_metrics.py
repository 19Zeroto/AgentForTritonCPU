#!/usr/bin/env python3
"""Extract fusion throughput metrics from FlagGems benchmark record logs.

Usage:
  python3 fusion_metrics.py report --logs <record-log-or-run-dir>
  python3 fusion_metrics.py update-report --logs <run-dir> --ops <op> \
      --output-dir <existing-report-dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


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

TARGET_TFLOPS = {
    ("SME", "float32"): 2.52631579,
    ("SME", "bfloat16"): 5.05263158,
    ("SVE", "float32"): 0.71578947,
}
DEFAULT_PIPELINE_BY_CATEGORY = {"compute": "SME", "memory": "SVE"}
OP_ALIASES = {"skip_layernorm": "skip_layer_norm"}
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
    "rwkv_ka_fusion": "fused/rwkv_ka_fusion.py",
    "rwkv_mm_sparsity": "fused/rwkv_mm_sparsity.py",
}


class FusionMetricError(ValueError):
    """A benchmark record cannot be converted into a throughput report."""


def canonical_op(name: Any) -> str:
    value = str(name).removeprefix("flag_gems.")
    return OP_ALIASES.get(value, value)


def canonical_dtype(value: Any) -> str:
    text = str(value).removeprefix("torch.").lower()
    aliases = {
        "fp32": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
        "half": "float16",
    }
    return aliases.get(text, text)


def _num(value: Any) -> float | None:
    try:
        result = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return result if result is not None and math.isfinite(result) else None


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _json_lines(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"\{.*\}", line)
        if not match:
            continue
        try:
            record = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "op_name" in record and "result" in record:
            yield record


def log_files(logs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for supplied in logs:
        path = supplied.expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("records/*.log")))
            files.extend(sorted(path.glob("raw/*.log")))
            files.extend(sorted(path.glob("*.log")))
        elif path.is_file():
            files.append(path)
    return list(dict.fromkeys(files))


def read_rows(
    logs: Iterable[Path], ops: set[str] | None = None
) -> list[dict[str, Any]]:
    selected = {canonical_op(op) for op in ops} if ops else None
    rows: list[dict[str, Any]] = []
    for path in log_files(logs):
        for record in _json_lines(path):
            op_name = canonical_op(record["op_name"])
            if selected and op_name not in selected:
                continue
            for item in record.get("result", []):
                if not isinstance(item, dict):
                    continue
                row = dict(item)
                row.update(
                    {
                        "op_name": op_name,
                        "dtype": canonical_dtype(record.get("dtype", "")),
                        "mode": record.get("mode", "N/A"),
                        "level": record.get("level", "N/A"),
                        "record_path": str(path),
                    }
                )
                rows.append(row)
    return rows


def _manifest_candidates(logs: Iterable[Path]) -> Iterable[Path]:
    for supplied in logs:
        path = supplied.expanduser().resolve()
        root = path if path.is_dir() else path.parent
        if root.name in {"records", "raw", "stdout"}:
            root = root.parent
        manifest = root / "run.json"
        if manifest.is_file():
            yield manifest


def _runtime_facts(logs: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    facts: dict[str, Any] = {}
    expected: list[dict[str, Any]] = []
    for manifest in dict.fromkeys(_manifest_candidates(logs)):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        common = {
            key: data.get(key, "N/A") for key in ("omp_threads", "warmup", "iter")
        }
        numa = f"cpu={data.get('cpu_node', 'N/A')};mem={data.get('mem_node', 'N/A')}"
        for item in data.get("operators", []):
            op_name = canonical_op(item.get("op_name", ""))
            op_facts = {
                **common,
                "numa": numa,
                "stdout": item.get("stdout", "N/A"),
                "record_log": item.get("record_log", "N/A"),
                "status": item.get("status", "N/A"),
            }
            facts[op_name] = op_facts
            expected.append({"op_name": op_name, **op_facts})
    return facts, expected


def _row_category(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    tflops = _num(row.get("tflops"))
    gbps = _num(row.get("gbps"))
    if tflops is not None and gbps is not None:
        return None, "record contains both tflops and gbps"
    if tflops is not None:
        return "compute", None
    if gbps is not None:
        return "memory", None
    return None, "record contains neither tflops nor gbps"


def _pipeline(category: str, override: str | None) -> str:
    return override or DEFAULT_PIPELINE_BY_CATEGORY[category]


def _target_tflops(
    pipeline: str, dtype: str, override: float | None
) -> float | None:
    return override if override is not None else TARGET_TFLOPS.get((pipeline, dtype))


def _shape_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _report_rows(
    rows: list[dict[str, Any]],
    pipeline: str | None,
    target_tflops: float | None,
    target_gbps: float,
    runtime: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    compute: list[dict[str, Any]] = []
    memory: list[dict[str, Any]] = []
    for row in rows:
        category, reason = _row_category(row)
        if row.get("error_msg") or category is None:
            continue
        gems_latency = _num(row.get("latency"))
        torch_latency = _num(row.get("latency_base"))
        if gems_latency is None or gems_latency <= 0:
            continue
        facts = runtime.get(row["op_name"], {})
        common = {
            "源码文件": SOURCE_FILES.get(row["op_name"], "N/A"),
            "pipeline": _pipeline(category, pipeline),
            "OMP线程数": facts.get("omp_threads", "N/A"),
            "warmup": facts.get("warmup", "N/A"),
            "iter": facts.get("iter", "N/A"),
            "benchmark模式": row.get("mode", "N/A"),
            "备注": reason or "",
        }
        speedup = _num(row.get("speedup"))
        if speedup is None and torch_latency is not None:
            speedup = torch_latency / gems_latency
        if category == "compute":
            measured = _num(row.get("tflops"))
            assert measured is not None
            logical_flops = measured * gems_latency * 1e9
            target = _target_tflops(common["pipeline"], row["dtype"], target_tflops)
            expected = logical_flops / target / 1e9 if target else None
            compute.append(
                {
                    "算子名称": row["op_name"],
                    "shape": _shape_text(row.get("shape_detail")),
                    "dtype": row["dtype"],
                    "Gems延迟(ms)": gems_latency,
                    "Torch延迟(ms)": torch_latency,
                    "目标算力下期望延迟(ms)": expected,
                    "目标达成率(%)": expected / gems_latency * 100 if expected else None,
                    "逻辑计算量(GFLOPs)": logical_flops / 1e9,
                    "实测算力(TFLOPS)": measured,
                    "目标算力(TFLOPS)": target,
                    "实测/期望延迟倍率": gems_latency / expected if expected else None,
                    "Torch Speedup": speedup,
                    **common,
                }
            )
        else:
            measured = _num(row.get("gbps"))
            assert measured is not None
            logical_bytes = measured * gems_latency * 1e6
            torch_gbps = _num(row.get("gbps_base"))
            if torch_gbps is None and torch_latency:
                torch_gbps = logical_bytes / torch_latency / 1e6
            expected = logical_bytes / target_gbps / 1e6 if target_gbps else None
            memory.append(
                {
                    "算子名称": row["op_name"],
                    "shape": _shape_text(row.get("shape_detail")),
                    "dtype": row["dtype"],
                    "Gems延迟(ms)": gems_latency,
                    "Torch延迟(ms)": torch_latency,
                    "目标带宽下期望延迟(ms)": expected,
                    "目标达成率(%)": expected / gems_latency * 100 if expected else None,
                    "逻辑数据量(GB)": logical_bytes / 1e9,
                    "逻辑带宽(GB/s)": measured,
                    "目标带宽(GB/s)": target_gbps,
                    "Torch逻辑带宽(GB/s)": torch_gbps,
                    "Speedup": speedup,
                    **common,
                }
            )
    return compute, memory


def _coverage_rows(
    rows: list[dict[str, Any]],
    pipeline: str | None,
    runtime: Mapping[str, Any],
    expected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    present: set[str] = set()
    for row in rows:
        op_name = row["op_name"]
        present.add(op_name)
        category, metric_reason = _row_category(row)
        error = row.get("error_msg")
        latency = _num(row.get("latency"))
        facts = runtime.get(op_name, {})
        if error:
            status, reason = "INVALID_LOG", str(error)
        elif category is None:
            status, reason = "METRIC_MISSING", metric_reason or "metric missing"
        elif latency is None or latency <= 0:
            status, reason = "INVALID_LOG", "invalid Gems latency"
        else:
            status, reason = "READY", ""
        result.append(
            {
                "算子名称": op_name,
                "类别": category or "N/A",
                "状态": status,
                "原因": reason,
                "测试文件": "N/A",
                "marker": "N/A",
                "pipeline": _pipeline(category, pipeline) if category else "N/A",
                "NUMA节点": facts.get("numa", "N/A"),
                "stdout": facts.get("stdout", "N/A"),
                "record log": row.get("record_path", facts.get("record_log", "N/A")),
            }
        )
    for item in expected:
        if item["op_name"] in present:
            continue
        result.append(
            {
                "算子名称": item["op_name"],
                "类别": "N/A",
                "状态": "LOG_MISSING",
                "原因": f"benchmark status: {item.get('status', 'N/A')}",
                "测试文件": "N/A",
                "marker": "N/A",
                "pipeline": "N/A",
                "NUMA节点": item.get("numa", "N/A"),
                "stdout": item.get("stdout", "N/A"),
                "record log": item.get("record_log", "N/A"),
            }
        )
    return result


def _write_csv(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _publish_snapshot(output_dir: Path, snapshot: Path, names: Iterable[str]) -> None:
    for name in names:
        temporary = output_dir / f".{name}.tmp"
        shutil.copyfile(snapshot / name, temporary)
        temporary.replace(output_dir / name)


def _default_output_dir(logs: list[Path]) -> Path:
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


def _workload_key(row: Mapping[str, Any]) -> str:
    return json.dumps(
        [row["op_name"], row["dtype"], row.get("shape_detail")],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_report(
    output_dir: Path,
    compute: list[dict[str, Any]],
    memory: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    snapshot.mkdir(parents=True, exist_ok=True)
    _write_csv(snapshot / "fused_compute_tflops.csv", COMPUTE_HEADER, compute)
    _write_csv(snapshot / "fused_non_compute_bandwidth.csv", MEMORY_HEADER, memory)
    _write_csv(snapshot / "coverage.csv", COVERAGE_HEADER, coverage)
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


def report(
    logs: list[Path],
    output_dir: Path,
    pipeline: str | None,
    target_tflops: float | None,
    target_gbps: float,
    ops: set[str] | None = None,
) -> Path:
    records = log_files(logs)
    if not records:
        raise FusionMetricError("no record logs found; inspect records/ and stdout/")
    rows = read_rows(records, ops)
    if not rows:
        raise FusionMetricError("record logs contain no benchmark records")
    runtime, expected = _runtime_facts(logs)
    compute, memory = _report_rows(
        rows, pipeline, target_tflops, target_gbps, runtime
    )
    coverage = _coverage_rows(rows, pipeline, runtime, expected)
    manifest = {
        "generated_at": _timestamp(),
        "pipeline": pipeline or "auto",
        "source_logs": [str(path) for path in logs],
        "rows": {"compute": len(compute), "memory": len(memory)},
    }
    return _write_report(output_dir, compute, memory, coverage, manifest)


def update_report(
    logs: list[Path],
    ops: set[str],
    output_dir: Path,
    pipeline: str | None,
    target_tflops: float | None,
    target_gbps: float,
) -> Path:
    selected = {canonical_op(op) for op in ops}
    rows = read_rows(logs, selected)
    found = {row["op_name"] for row in rows}
    missing = selected - found
    if missing:
        raise FusionMetricError(
            f"selected operators missing from logs: {', '.join(sorted(missing))}"
        )
    keys: set[str] = set()
    for row in rows:
        key = _workload_key(row)
        if key in keys:
            raise FusionMetricError(f"duplicate workload in update logs: {key}")
        keys.add(key)
    runtime, expected = _runtime_facts(logs)
    new_compute, new_memory = _report_rows(
        rows, pipeline, target_tflops, target_gbps, runtime
    )
    selected_expected = [
        item for item in expected if item["op_name"] in selected
    ]
    new_coverage = _coverage_rows(
        rows, pipeline, runtime, selected_expected
    )
    invalid = [row for row in new_coverage if row["状态"] != "READY"]
    if invalid:
        details = "; ".join(
            f"{row['算子名称']}: {row['状态']} ({row['原因']})"
            for row in invalid
        )
        raise FusionMetricError(
            f"selected operators are not ready for update: {details}"
        )

    def existing(name: str) -> list[dict[str, Any]]:
        path = output_dir / name
        if not path.is_file():
            return []
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    compute = [
        row
        for row in existing("fused_compute_tflops.csv")
        if canonical_op(row.get("算子名称", "")) not in selected
    ] + new_compute
    memory = [
        row
        for row in existing("fused_non_compute_bandwidth.csv")
        if canonical_op(row.get("算子名称", "")) not in selected
    ] + new_memory
    coverage = [
        row
        for row in existing("coverage.csv")
        if canonical_op(row.get("算子名称", "")) not in selected
    ] + new_coverage
    manifest = {
        "generated_at": _timestamp(),
        "pipeline": pipeline or "auto",
        "source_logs": [str(path) for path in logs],
        "updated_ops": sorted(selected),
        "rows": {"compute": len(new_compute), "memory": len(new_memory)},
    }
    return _write_report(output_dir, compute, memory, coverage, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("report", "update-report"):
        command = subparsers.add_parser(name)
        command.add_argument("--logs", nargs="+", type=Path, required=True)
        command.add_argument(
            "--output-dir",
            type=Path,
            help="Report directory (default: <run-dir>/report)",
        )
        command.add_argument("--pipeline", choices=("SME", "SVE"))
        command.add_argument("--target-tflops", type=float)
        command.add_argument("--target-gbps", type=float, default=30.72)
        if name == "update-report":
            command.add_argument("--ops", nargs="+", required=True)
    args = parser.parse_args(argv)
    output_dir = args.output_dir or _default_output_dir(args.logs)
    try:
        if args.command == "report":
            report(
                args.logs,
                output_dir,
                args.pipeline,
                args.target_tflops,
                args.target_gbps,
            )
        else:
            update_report(
                args.logs,
                set(args.ops),
                output_dir,
                args.pipeline,
                args.target_tflops,
                args.target_gbps,
            )
    except (OSError, KeyError, ValueError, FusionMetricError) as exc:
        parser.error(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
