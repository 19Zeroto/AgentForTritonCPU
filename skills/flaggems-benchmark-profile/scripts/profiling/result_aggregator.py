from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .flops import resolve_tflops
from .perf_wrapper import PerfStatRunner, derive_cache_metrics


class ResultAggregator:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        generated_at: str,
        config_file: str,
        system_info: dict[str, Any] | None = None,
        run_mode: str = "group",
    ):
        self.run_dir = Path(run_dir)
        self.generated_at = generated_at
        self.config_file = config_file
        self.system_info = system_info or {}
        self.run_mode = run_mode
        self.results: list[dict[str, Any]] = []

    def add_op_result(
        self,
        op_name: str,
        *,
        test_file: str,
        marker: str,
        status: str,
        returncode: int,
        wall_time_sec: float,
        compile_log_path: str | Path | None,
        cpu_memory_csv: str | Path | None,
        perf_csv: str | Path | None,
        perf_data_path: str | Path | None = None,
        perf_report_path: str | Path | None = None,
        perf_top_path: str | Path | None = None,
        record_log_path: str | Path | None,
        stdout_path: str | Path,
        command: list[str],
        note: str | None = None,
        warmup: int | None = None,
        iterations: int | None = None,
        mode: str | None = None,
        level: str | None = None,
        dtypes: list[str] | None = None,
        record: str | None = None,
        run_mode: str = "group",
        run_label: str | None = None,
        case_index: int | None = None,
        requested_shape: Any = None,
    ) -> dict[str, Any]:
        benchmark_results = parse_record_log(record_log_path)
        cases = flatten_benchmark_cases(benchmark_results)
        compilation = summarize_compile_log(compile_log_path)
        cpu_memory = summarize_cpu_memory(cpu_memory_csv)
        cache = summarize_perf(perf_csv)
        result = {
            "op_name": op_name,
            "test_file": test_file,
            "marker": marker,
            "status": status,
            "returncode": returncode,
            "note": note,
            "mode": mode,
            "level": level,
            "run_mode": run_mode,
            "run_label": run_label or op_name,
            "case_index": case_index,
            "requested_shape": requested_shape,
            "warmup": warmup,
            "iterations": iterations,
            "dtypes": dtypes,
            "record": record,
            "wall_time_sec": wall_time_sec,
            "artifacts": {
                "stdout": _relative(stdout_path, self.run_dir),
                "record_log": _relative(record_log_path, self.run_dir),
                "compile_log": _relative(compile_log_path, self.run_dir),
                "cpu_memory_csv": _relative(cpu_memory_csv, self.run_dir),
                "perf_csv": _relative(perf_csv, self.run_dir),
                "perf_data": _relative(perf_data_path, self.run_dir),
                "perf_report": _relative(perf_report_path, self.run_dir),
                "perf_top10": _relative(perf_top_path, self.run_dir),
            },
            "command": command,
            "compilation": compilation,
            "cpu_memory": cpu_memory,
            "cache": cache,
            "latency": summarize_latency_from_cases(cases),
            "tflops": summarize_tflops_from_cases(cases),
            "cases": attach_run_context_to_cases(
                cases,
                wall_time_sec=wall_time_sec,
                compilation=compilation,
                cpu_memory=cpu_memory,
                cache=cache,
                run_mode=run_mode,
                run_label=run_label or op_name,
                case_index=case_index,
                requested_shape=requested_shape,
            ),
            "benchmark_results": benchmark_results,
        }
        self.results.append(result)
        return result

    def write_summary(self, output_dir: str | Path | None = None) -> None:
        output_path = Path(output_dir) if output_dir is not None else self.run_dir
        summary = {
            "run_info": {
                "generated_at": self.generated_at,
                "config_file": self.config_file,
                "run_mode": self.run_mode,
                "system": self.system_info.get("summary", self.system_info),
            },
            "status_counts": _status_counts(self.results),
            "results": self.results,
        }
        (output_path / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        (output_path / "summary.md").write_text(
            render_summary_markdown(summary), encoding="utf-8"
        )


def parse_record_log(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    record_path = Path(path)
    if not record_path.exists():
        return []

    payloads: list[dict[str, Any]] = []
    for line in record_path.read_text(encoding="utf-8", errors="replace").splitlines():
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


def summarize_compile_log(path: str | Path | None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if path is not None:
        compile_path = Path(path)
        if compile_path.exists():
            for line in compile_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)

    compile_times = [
        float(record["compile_ms"])
        for record in records
        if _is_number(record.get("compile_ms"))
    ]
    return {
        "total_compile_time_ms": sum(compile_times),
        "num_compilations": len(compile_times),
        "per_shape": records,
    }


def summarize_cpu_memory(path: str | Path | None) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    if path is not None and Path(path).exists():
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                parsed = {
                    key: float(value)
                    for key, value in row.items()
                    if key is not None and _is_number(value)
                }
                if parsed:
                    rows.append(parsed)

    cpu_values = [row.get("cpu_percent", 0.0) for row in rows]
    rss_values = [row.get("rss_mb", 0.0) for row in rows]
    vms_values = [row.get("vms_mb", 0.0) for row in rows]
    return {
        "samples": len(rows),
        "peak_cpu_pct": max(cpu_values) if cpu_values else None,
        "avg_cpu_pct": sum(cpu_values) / len(cpu_values) if cpu_values else None,
        "peak_rss_mb": max(rss_values) if rss_values else None,
        "avg_rss_mb": sum(rss_values) / len(rss_values) if rss_values else None,
        "peak_vms_mb": max(vms_values) if vms_values else None,
        "avg_vms_mb": sum(vms_values) / len(vms_values) if vms_values else None,
    }


def summarize_perf(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    perf_result = PerfStatRunner.parse_output(path)
    return derive_cache_metrics(perf_result.events)


def flatten_benchmark_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for record in records:
        benchmark_op = record.get("op_name")
        dtype = record.get("dtype")
        mode = record.get("mode")
        level = record.get("level")
        for index, metric in enumerate(record.get("result", [])):
            shape_detail = metric.get("shape_detail")
            status = "failed" if metric.get("error_msg") else "success"
            case_name = build_case_name(benchmark_op, dtype, shape_detail, index)
            tflops_info = resolve_tflops(
                benchmark_op,
                shape_detail,
                metric.get("latency"),
                metric.get("tflops"),
            )
            cases.append(
                {
                    "case_name": case_name,
                    "benchmark_op_name": benchmark_op,
                    "dtype": dtype,
                    "mode": mode,
                    "level": level,
                    "shape_detail": shape_detail,
                    "status": status,
                    "latency_base": _float_or_none(metric.get("latency_base")),
                    "latency": _float_or_none(metric.get("latency")),
                    "speedup": _float_or_none(metric.get("speedup")),
                    "gbps_base": _float_or_none(metric.get("gbps_base")),
                    "gbps": _float_or_none(metric.get("gbps")),
                    "tflops": tflops_info.tflops,
                    "flops": tflops_info.flops,
                    "tflops_source": tflops_info.source,
                    "flops_formula_id": tflops_info.formula_id,
                    "utilization": _float_or_none(metric.get("utilization")),
                    "accuracy": _float_or_none(metric.get("accuracy")),
                    "compared_speedup": _float_or_none(
                        metric.get("compared_speedup")
                    ),
                    "error_msg": metric.get("error_msg"),
                    "raw_metrics": metric,
                }
            )
    return cases


def attach_run_context_to_cases(
    cases: list[dict[str, Any]],
    *,
    wall_time_sec: float,
    compilation: dict[str, Any],
    cpu_memory: dict[str, Any],
    cache: dict[str, Any],
    run_mode: str,
    run_label: str,
    case_index: int | None,
    requested_shape: Any,
) -> list[dict[str, Any]]:
    total_compile_ms = compilation.get("total_compile_time_ms")
    peak_cpu_pct = cpu_memory.get("peak_cpu_pct")
    peak_rss_mb = cpu_memory.get("peak_rss_mb")
    case_rows = []
    for case in cases:
        enriched = dict(case)
        enriched["run_context"] = {
            "scope": run_mode,
            "run_label": run_label,
            "case_index": case_index,
            "requested_shape": requested_shape,
            "wall_time_sec": wall_time_sec,
            "total_compile_time_ms": total_compile_ms,
            "peak_cpu_pct": peak_cpu_pct,
            "peak_rss_mb": peak_rss_mb,
            "cache": cache,
        }
        case_rows.append(enriched)
    return case_rows


def summarize_latency_from_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(case["latency"])
        for case in cases
        if _is_number(case.get("latency"))
    ]
    per_shape = [
        {
            "case_name": case.get("case_name"),
            "op_name": case.get("benchmark_op_name"),
            "dtype": case.get("dtype"),
            "shape_detail": case.get("shape_detail"),
            "latency": case.get("latency"),
            "latency_base": case.get("latency_base"),
            "speedup": case.get("speedup"),
        }
        for case in cases
        if _is_number(case.get("latency"))
    ]
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "per_shape": per_shape,
    }


def summarize_tflops_from_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    per_shape = [
        {
            "case_name": case.get("case_name"),
            "op_name": case.get("benchmark_op_name"),
            "dtype": case.get("dtype"),
            "shape_detail": case.get("shape_detail"),
            "tflops": case.get("tflops"),
            "flops": case.get("flops"),
            "tflops_source": case.get("tflops_source"),
            "flops_formula_id": case.get("flops_formula_id"),
        }
        for case in cases
        if _is_number(case.get("tflops")) and float(case.get("tflops")) != 0.0
    ]
    values = [float(case["tflops"]) for case in per_shape]
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "per_shape": per_shape,
    }


def build_case_name(
    op_name: Any, dtype: Any, shape_detail: Any, index: int
) -> str:
    shape_label = format_shape_detail(shape_detail)
    parts = [str(op_name or "unknown_op"), str(dtype or "unknown_dtype")]
    if shape_label != "N/A":
        parts.append(shape_label)
    else:
        parts.append(f"case_{index}")
    return " | ".join(parts)


def render_summary_markdown(summary: dict[str, Any]) -> str:
    run_info = summary["run_info"]
    lines = [
        "# FlagGems Profiling Summary",
        "",
        f"- Generated at: {run_info.get('generated_at')}",
        f"- Config: {run_info.get('config_file')}",
        f"- Run mode: {run_info.get('run_mode', 'group')}",
        "",
        "## Overall",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(summary.get("status_counts", {}).items()):
        lines.append(f"| {status} | {count} |")

    lines.extend(
        [
            "",
            "## Operators",
            "",
            "| Operator | Run Label | Scope | Status | Wall Time (s) | Compile (ms) | Peak CPU % | "
            "Peak RSS MB | Mean TFLOPS | Note |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in summary.get("results", []):
        row = (
            "| {op} | {label} | {scope} | {status} | {wall} | {compile_ms} | {cpu} | {rss} | "
            "{tflops} | {note} |"
        )
        lines.append(
            row.format(
                op=_escape_md(result.get("op_name")),
                label=_escape_md(result.get("run_label") or result.get("op_name")),
                scope=_escape_md(result.get("run_mode") or "group"),
                status=_escape_md(result.get("status")),
                wall=_fmt(result.get("wall_time_sec")),
                compile_ms=_fmt(
                    result.get("compilation", {}).get("total_compile_time_ms")
                ),
                cpu=_fmt(result.get("cpu_memory", {}).get("peak_cpu_pct")),
                rss=_fmt(result.get("cpu_memory", {}).get("peak_rss_mb")),
                tflops=_fmt(result.get("tflops", {}).get("mean")),
                note=_escape_md(result.get("note") or ""),
            )
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Operator | Run Label | Scope | Case | Status | DType | Shape | Torch Latency (ms) | "
            "Gems Latency (ms) | Speedup | Torch GBPS | Gems GBPS | TFLOPS | "
            "Profile Wall (s) | Profile Compile (ms) | Peak CPU % | Peak RSS MB | L1 Miss Rate | "
            "LLC Miss Rate | IPC | Instructions | Cycles | Error |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in summary.get("results", []):
        for case in result.get("cases", []):
            context = case.get("run_context", {})
            cache = context.get("cache", {})
            lines.append(
                "| {op} | {label} | {scope} | {case} | {status} | {dtype} | {shape} | {base} | "
                "{latency} | {speedup} | {gbps_base} | {gbps} | {tflops} | "
                "{profile_wall} | {compile_ms} | {cpu} | {rss} | {l1_miss_rate} | "
                "{llc_miss_rate} | {ipc} | {instructions} | {cycles} | "
                "{error} |".format(
                    op=_escape_md(result.get("op_name")),
                    label=_escape_md(
                        context.get("run_label") or result.get("run_label")
                    ),
                    scope=_escape_md(context.get("scope") or result.get("run_mode")),
                    case=_escape_md(case.get("case_name")),
                    status=_escape_md(case.get("status")),
                    dtype=_escape_md(case.get("dtype")),
                    shape=_escape_md(format_shape_detail(case.get("shape_detail"))),
                    base=_fmt(case.get("latency_base")),
                    latency=_fmt(case.get("latency")),
                    speedup=_fmt(case.get("speedup")),
                    gbps_base=_fmt(case.get("gbps_base")),
                    gbps=_fmt(case.get("gbps")),
                    tflops=_fmt(case.get("tflops")),
                    profile_wall=_fmt(context.get("wall_time_sec")),
                    compile_ms=_fmt(context.get("total_compile_time_ms")),
                    cpu=_fmt(context.get("peak_cpu_pct")),
                    rss=_fmt(context.get("peak_rss_mb")),
                    l1_miss_rate=_fmt(cache.get("L1_dcache_miss_rate")),
                    llc_miss_rate=_fmt(cache.get("LLC_miss_rate")),
                    ipc=_fmt(cache.get("ipc")),
                    instructions=_fmt(cache.get("instructions")),
                    cycles=_fmt(cache.get("cycles")),
                    error=_escape_md(case.get("error_msg") or ""),
                )
            )
    return "\n".join(lines) + "\n"


def determine_status(
    returncode: int, benchmark_results: list[dict[str, Any]], output: str
) -> tuple[str, str | None]:
    if returncode == 0 and benchmark_results:
        return "passed", None
    if returncode != 0 and benchmark_results:
        return "partial", extract_failure_note(output, returncode)
    lowered = output.lower()
    if returncode == 0 and "skipped" in lowered:
        return "skipped", "benchmark skipped by pytest conditions"
    if returncode != 0:
        return "failed", extract_failure_note(output, returncode)
    return "no-data", "no benchmark JSON records were produced"


def extract_failure_note(output: str, returncode: int) -> str:
    interesting_prefixes = (
        "ModuleNotFoundError:",
        "ImportError:",
        "RuntimeError:",
        "ValueError:",
        "AssertionError:",
        "Failed:",
    )
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(interesting_prefixes):
            return stripped[:240]
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped:
            return f"pytest exit code {returncode}: {stripped[:240]}"
    return f"pytest exit code {returncode}"


def _relative(path: str | Path | None, root: Path) -> str | None:
    if path is None:
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    try:
        return str(path_obj.relative_to(root))
    except ValueError:
        return str(path_obj)


def _status_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _float_or_none(value: Any) -> float | None:
    return float(value) if _is_number(value) else None


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if _is_number(value):
        return f"{float(value):.3f}"
    return str(value)


def _escape_md(value: Any) -> str:
    return str(value).replace("|", "/")


def format_shape_detail(shape_detail: Any) -> str:
    if shape_detail is None:
        return "N/A"
    return json.dumps(shape_detail, ensure_ascii=False, default=str)
