#!/usr/bin/env python3
"""Run selected FlagGems benchmarks serially, collect logs, and export a report.

The benchmark commands remain isolated from post-processing.  Once the run is
complete, ``fusion_metrics.py report`` consumes the collected logs and writes a
report below the run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required by run_fusion_benchmarks.py") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
AGENT_DIR = Path(
    os.environ.get("AGENT_DIR", str(SKILL_DIR.parents[2]))
).expanduser().resolve()
REPO_DIR = Path(
    os.environ.get("TRITON_REPO_DIR", str(AGENT_DIR / "triton-cpu"))
).expanduser().resolve()
BENCHMARK_DIR = REPO_DIR / "FlagGems" / "benchmark"
DEFAULT_SHAPE_FILE = BENCHMARK_DIR / "core_shapes.yaml"
CASES_PATH = SCRIPT_DIR / "benchmark_cases.yaml"
FUSION_METRICS_SCRIPT = SCRIPT_DIR / "fusion_metrics.py"
DEFAULT_OUTPUT = AGENT_DIR / "logs" / "fusion_metrics" / "benchmark_runs"
INJECTION_VARS = (
    "FLAGGEMS_FUSION_METRICS_CONFIG",
    "FLAGGEMS_FUSION_METRICS_KEY",
    "TRITON_SHARED_FORCE_SME_PIPELINE",
    "TRITON_SHARED_FORCE_SVE_PIPELINE",
)

# Keep the suite definitions explicit and reviewable. Pass --ops to run a
# custom subset, or select one of these categories with --suite.
COMPUTE_OPS = [
    "flash_attention_forward",
    "flash_attn_varlen_func",
    "flash_mla",
    "flash_mla_sparse_fwd",
    "sparse_mla_fwd_interface",
    "rwkv_mm_sparsity",
]
MEMORY_OPS = [
    "apply_repetition_penalties",
    "apply_rotary_pos_emb",
    "concat_and_cache_mla",
    "cross_entropy_loss",
    "dgeglu",
    "dreglu",
    "fused_add_rms_norm",
    "geglu",
    "gelu_and_mul",
    "get_scheduler_metadata",
    "instance_norm",
    "moe_sum",
    "reglu",
    "reshape_and_cache",
    "reshape_and_cache_flash",
    "rwkv_ka_fusion",
    "silu_and_mul",
    "silu_and_mul_out",
    "skip_layer_norm",
    # "weight_norm",
    # "topk_softmax",
    # "moe_align_block_size_triton",
]
SUITE_OPS = {
    "compute": COMPUTE_OPS,
    "memory": MEMORY_OPS,
    "all": COMPUTE_OPS + MEMORY_OPS,
}


def _parse_ranges(spec: str, label: str) -> list[int]:
    """Parse numactl-style numeric lists such as ``300-331,400,402-403``."""
    values: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            raise ValueError(f"empty item in {label}: {spec!r}")
        bounds = item.split("-")
        if len(bounds) > 2 or not all(part.isdigit() for part in bounds):
            raise ValueError(f"invalid {label} range: {item!r}")
        start = int(bounds[0])
        end = int(bounds[-1])
        if end < start:
            raise ValueError(f"descending {label} range: {item!r}")
        values.extend(range(start, end + 1))
    return list(dict.fromkeys(values))


def _format_ranges(values: list[int]) -> str:
    """Format CPU ids as compact comma-separated ranges for numactl."""
    if not values:
        raise ValueError("CPU binding cannot be empty")
    ordered = sorted(set(values))
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _cpulist(node: int) -> list[int]:
    path = Path(f"/sys/devices/system/node/node{node}/cpulist")
    try:
        return _parse_ranges(path.read_text().strip(), f"NUMA node {node} CPU list")
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read CPU list for NUMA node {node}") from exc


def _has_memory(node: int) -> bool:
    try:
        lines = Path(f"/sys/devices/system/node/node{node}/meminfo").read_text()
    except OSError:
        return False
    for line in lines.splitlines():
        if "MemTotal:" not in line:
            continue
        try:
            return int(line.split("MemTotal:", 1)[1].split()[0]) > 0
        except (IndexError, ValueError):
            return False
    return False


def _validate_binding(
    cpu_nodes_spec: str,
    mem_nodes_spec: str,
    omp_threads: int,
    cpu_spec: str | None,
) -> tuple[list[int], str]:
    cpu_nodes = _parse_ranges(cpu_nodes_spec, "CPU NUMA node")
    mem_nodes = _parse_ranges(mem_nodes_spec, "memory NUMA node")
    if not cpu_nodes:
        raise ValueError("CPU NUMA node selection cannot be empty")
    if not mem_nodes:
        raise ValueError("memory NUMA node selection cannot be empty")
    if omp_threads > 32:
        raise ValueError("OMP threads must be <= 32")
    available: list[int] = []
    for node in cpu_nodes:
        available.extend(_cpulist(node))
    available = list(dict.fromkeys(available))
    if not available:
        raise ValueError("selected CPU NUMA nodes have no CPUs")
    missing_memory = [node for node in mem_nodes if not _has_memory(node)]
    if missing_memory:
        raise ValueError(
            f"memory NUMA nodes have no available memory: {missing_memory}"
        )

    selected = (
        _parse_ranges(cpu_spec, "CPU list")
        if cpu_spec
        else available[:omp_threads]
    )
    if not selected:
        raise ValueError("CPU binding cannot be empty")
    if len(selected) < omp_threads:
        raise ValueError("CPU list contains fewer CPUs than OMP threads")
    if not set(selected).issubset(available):
        raise ValueError("CPU list is not contained in the selected CPU NUMA nodes")
    return selected, _format_ranges(selected)


def _load_cases() -> dict[str, Any]:
    raw = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _record_path(work: Path) -> Path | None:
    logs = sorted(path for path in work.glob("*.log") if path.is_file())
    return logs[-1] if logs else None


def _print_status(op_name: str, status: str, returncode: int | None = None) -> None:
    suffix = "" if returncode is None else f" (returncode={returncode})"
    print(f"[{status.upper()}] {op_name}{suffix}")


def _generate_report(root: Path) -> tuple[int, str]:
    """Run the post-processor after benchmark execution has finished."""
    command = [
        sys.executable,
        str(FUSION_METRICS_SCRIPT),
        "report",
        "--logs",
        str(root),
    ]
    print(f"report command: {shlex.join(command)}")
    report_log = root / "stdout" / "fusion_metrics_report.log"
    report_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as exc:
        report_log.write_text(str(exc) + "\n", encoding="utf-8")
        print(f"[REPORT_FAILED] {exc}")
        return 127, str(report_log.relative_to(root))

    report_log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode == 0:
        print(f"[REPORT] {root / 'report'}")
    else:
        print(f"[REPORT_FAILED] returncode={completed.returncode}")
        print(f"report output: {report_log}")
    return completed.returncode, str(report_log.relative_to(root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ops",
        nargs="+",
        help="Custom operator subset; mutually exclusive with --suite.",
    )
    parser.add_argument(
        "--suite",
        choices=tuple(SUITE_OPS),
        default="all",
        help="Run the compute, memory, or all default operator list.",
    )
    parser.add_argument(
        "--mode", choices=("kernel", "operator", "wrapper"), default="operator"
    )
    parser.add_argument(
        "--level", choices=("core", "comprehensive"), default="core"
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument(
        "--shape-file",
        type=Path,
        default=None,
        help="Optional custom shape file; the benchmark default is used otherwise.",
    )
    parser.add_argument("--dtypes", nargs="+")
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--cpu-node", default="0", help="NUMA node list, e.g. 0 or 0,1")
    parser.add_argument(
        "--mem-node", default="0", help="Memory NUMA node list, e.g. 20,22,23,27"
    )
    parser.add_argument(
        "--cpu-list",
        help="Physical CPU list/ranges, e.g. 300-331 or 300-331,400-403",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.ops and args.suite != "all":
        parser.error("--ops and --suite cannot be used together")
    selected_ops = args.ops or SUITE_OPS[args.suite]
    cases = _load_cases()
    unknown = [op for op in selected_ops if op not in cases]
    if unknown:
        parser.error(f"unknown operators: {', '.join(unknown)}")
    if args.warmup < 0 or args.iter <= 0:
        parser.error("warmup must be >= 0 and iter must be > 0")
    if args.omp_threads <= 0:
        parser.error("OMP threads must be > 0")
    for name in INJECTION_VARS:
        if os.environ.get(name):
            parser.error(f"injection/pipeline environment variable is set: {name}")

    try:
        cpus, cpu_bind_spec = _validate_binding(
            args.cpu_node, args.mem_node, args.omp_threads, args.cpu_list
        )
    except ValueError as exc:
        parser.error(str(exc))

    shape_file = (args.shape_file or DEFAULT_SHAPE_FILE).expanduser().resolve()
    if not FUSION_METRICS_SCRIPT.is_file():
        parser.error(f"report script not found: {FUSION_METRICS_SCRIPT}")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = args.output_dir.expanduser().resolve() / run_id
    records = root / "records"
    stdout = root / "stdout"
    work_root = root / "work"
    if not args.dry_run:
        for path in (records, stdout, work_root):
            path.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "suite": "custom" if args.ops else args.suite,
        "mode": args.mode,
        "level": args.level,
        "warmup": args.warmup,
        "iter": args.iter,
        "shape_file": str(shape_file),
        "omp_threads": args.omp_threads,
        "cpu_node": args.cpu_node,
        "mem_node": args.mem_node,
        "cpu_list": cpu_bind_spec,
        "cpu_count": len(cpus),
        "operators": [],
    }
    failed = False

    for index, op in enumerate(selected_ops):
        case = cases[op]
        if args.dtypes and len(args.dtypes) == len(selected_ops):
            dtype = args.dtypes[index]
        elif args.dtypes:
            dtype = args.dtypes[0]
        else:
            dtype = case["dtype"]

        test_file = (BENCHMARK_DIR / case["test_file"]).resolve()
        if not test_file.is_file():
            parser.error(f"benchmark test file not found: {test_file}")
        operation_work = work_root / op
        execution_work = Path(tempfile.gettempdir()) / f"fgm-{run_id}-{index}"
        command = [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "-q",
            "--tb=no",
            "-m",
            str(case["marker"]),
            "--mode",
            args.mode,
            "--level",
            args.level,
            "--warmup",
            str(args.warmup),
            "--iter",
            str(args.iter),
            "--dtypes",
            dtype,
            "--record",
            "log",
        ]
        if args.shape_file is not None:
            command.extend(["--shape_file", "shape_file.yaml"])

        prefix = [
            "numactl",
            f"--cpunodebind={args.cpu_node}",
            f"--membind={args.mem_node}",
            f"--physcpubind={cpu_bind_spec}",
        ]
        full_command = prefix + command
        entry: dict[str, Any] = {
            "op_name": op,
            "dtype": dtype,
            "command": full_command,
            "status": "dry-run" if args.dry_run else "pending",
            "returncode": None,
            "record_log": f"records/{op}.log",
            "stdout": f"stdout/{op}.log",
            "work_dir": f"work/{op}",
        }
        manifest["operators"].append(entry)
        print(shlex.join(full_command))
        if args.dry_run:
            _print_status(op, "dry-run")
            continue

        execution_work.mkdir(parents=True, exist_ok=True)
        operation_work.parent.mkdir(parents=True, exist_ok=True)
        operation_work.symlink_to(execution_work, target_is_directory=True)
        benchmark_link = execution_work / "benchmark"
        if not benchmark_link.exists():
            benchmark_link.symlink_to(BENCHMARK_DIR, target_is_directory=True)
        if args.shape_file is not None:
            shape_link = execution_work / "shape_file.yaml"
            if not shape_link.exists():
                shape_link.symlink_to(shape_file)
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(args.omp_threads)
        env["PYTHONPATH"] = str(BENCHMARK_DIR) + os.pathsep + env.get(
            "PYTHONPATH", ""
        )
        try:
            completed = subprocess.run(
                full_command,
                cwd=execution_work,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout or None,
                check=False,
            )
            (stdout / f"{op}.log").write_text(completed.stdout, encoding="utf-8")
            entry["returncode"] = completed.returncode
            record = _record_path(execution_work)
            if record:
                shutil.move(str(record), str(records / f"{op}.log"))
            entry["status"] = "passed" if completed.returncode == 0 else "failed"
            failed |= completed.returncode != 0
            _print_status(op, entry["status"], completed.returncode)
        except subprocess.TimeoutExpired as exc:
            (stdout / f"{op}.log").write_text(exc.stdout or "", encoding="utf-8")
            entry["status"] = "timeout"
            failed = True
            _print_status(op, "timeout")

    root.mkdir(parents=True, exist_ok=True)
    (root / "run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report_returncode = None
    report_log = None
    if not args.dry_run:
        report_returncode, report_log = _generate_report(root)
        manifest["report"] = {
            "status": "passed" if report_returncode == 0 else "failed",
            "returncode": report_returncode,
            "path": "report",
            "stdout": report_log,
        }
        (root / "run.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"run artifacts: {root}")
    print("benchmark summary:")
    for entry in manifest["operators"]:
        _print_status(entry["op_name"], entry["status"], entry["returncode"])
    if report_returncode not in (None, 0):
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
