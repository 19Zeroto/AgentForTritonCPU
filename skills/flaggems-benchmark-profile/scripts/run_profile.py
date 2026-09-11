from __future__ import annotations

import argparse
import ast
import atexit
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from profiling.config import RunConfig, TestEntry
    from profiling.monitor import ResourceMonitor
    from profiling.perf_wrapper import PerfRecordRunner, PerfStatRunner
    from profiling.result_aggregator import (
        ResultAggregator,
        determine_status,
        parse_record_log,
    )
    from profiling.system_info import write_system_info
except ImportError:  # pragma: no cover - supports python -m benchmark.run_profile
    from benchmark.profiling.config import RunConfig, TestEntry
    from benchmark.profiling.monitor import ResourceMonitor
    from benchmark.profiling.perf_wrapper import PerfRecordRunner, PerfStatRunner
    from benchmark.profiling.result_aggregator import (
        ResultAggregator,
        determine_status,
        parse_record_log,
    )
    from benchmark.profiling.system_info import write_system_info


SCRIPT_DIR = Path(__file__).resolve().parent
PROFILING_DIR = SCRIPT_DIR / "profiling"
DEFAULT_CONFIG = SCRIPT_DIR / "profiling_config.yaml"
AGENTFORTRITONCPU_DIR = SCRIPT_DIR.parent.parent.parent
AGENT_DIR = Path(
    os.environ.get("AGENT_DIR", Path.home() / "agent")
).expanduser().resolve()
TRITON_CPU_ROOT = Path(
    os.environ.get(
        "TRITON_REPO_DIR",
        os.environ.get("TRITON_CPU_ROOT", AGENT_DIR / "triton-cpu"),
    )
).expanduser().resolve()
FLAGGEMS_ROOT = TRITON_CPU_ROOT / "FlagGems"
BENCHMARK_DIR = FLAGGEMS_ROOT / "benchmark"
SRC_DIR = FLAGGEMS_ROOT / "src"
DEFAULT_OUTPUT_ROOT = (
    AGENT_DIR / "logs" / "AgentForTritonCPU" / "flaggems-benchmark-profile"
)
DEFAULT_METRICS = None

PRINT_LOCK = threading.Lock()
ACTIVE_PROCESSES_LOCK = threading.Lock()
ACTIVE_PROCESSES: dict[int, tuple[str, subprocess.Popen[str]]] = {}
INTERRUPT_EVENT = threading.Event()


@dataclass
class RunArtifacts:
    op_name: str
    test_file: str
    marker: str
    status: str
    returncode: int
    note: str | None
    wall_time_sec: float
    compile_log_path: Path | None
    cpu_memory_csv: Path | None
    perf_csv: Path | None
    perf_data_path: Path | None
    perf_report_path: Path | None
    perf_top_path: Path | None
    record_log_path: Path | None
    stdout_path: Path
    command: list[str]
    warmup: int
    iterations: int
    mode: str
    level: str
    dtypes: list[str] | None
    record: str
    run_mode: str = "group"
    run_label: str | None = None
    case_index: int | None = None
    requested_shape: Any = None


@dataclass
class CaseSpec:
    index: int
    shape: Any
    shape_desc: str | None
    shape_file: Path
    shape_file_arg: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FlagGems benchmarks with profiling data collection."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Profiling YAML config. Defaults to {DEFAULT_CONFIG}.",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        default=None,
        help="Markers or test names to run, for example: --tests silu_and_mul add.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every test entry listed in the profiling config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where profile_<timestamp> results are written.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run directory name under --output-dir.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of benchmark subprocesses to run concurrently.",
    )
    parser.add_argument(
        "--skip-perf",
        action="store_true",
        help="Skip perf stat cache metrics collection.",
    )
    parser.add_argument(
        "--perf-record",
        action="store_true",
        help=(
            "Run the benchmark under perf record and write perf.data, "
            "perf_report.txt, and perf_top10.txt into the op/case directory. "
            "This disables perf stat for that run."
        ),
    )
    parser.add_argument(
        "--perf-record-frequency",
        type=int,
        default=99,
        help="Sampling frequency passed to perf record -F. Defaults to 99.",
    )
    parser.add_argument(
        "--perf-record-call-graph",
        choices=["dwarf", "fp", "none"],
        default="dwarf",
        help="Call graph mode passed to perf record. Defaults to dwarf.",
    )
    parser.add_argument(
        "--skip-cpu-mem",
        action="store_true",
        help="Skip psutil CPU/memory sampling.",
    )
    parser.add_argument(
        "--warmup-override",
        type=int,
        default=None,
        help="Override the resolved warmup count for all selected tests.",
    )
    parser.add_argument(
        "--iter-override",
        type=int,
        default=None,
        help="Override the resolved iteration count for all selected tests.",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=None,
        help="Dtype filter forwarded to pytest, for example: --dtypes float32.",
    )
    parser.add_argument(
        "--record",
        choices=["log", "none"],
        default="log",
        help="Benchmark record mode forwarded to pytest. Defaults to log.",
    )
    parser.add_argument(
        "--run-mode",
        choices=["group", "case"],
        default="group",
        help=(
            "group runs one pytest process per operator. case first discovers shapes "
            "and then runs one pytest process per shape so perf/compile counters are "
            "collected per shape."
        ),
    )
    parser.add_argument(
        "--case-indices",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Only with --run-mode case: run selected zero-based shape indices after "
            "shape discovery, for quick focused checks."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing benchmarks or writing results.",
    )
    args = parser.parse_args()
    if args.all and args.tests:
        parser.error("--all cannot be used together with --tests")
    if not args.all and not args.tests:
        parser.error("specify --all or --tests")
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    if args.warmup_override is not None and args.warmup_override <= 0:
        parser.error("--warmup-override must be a positive integer")
    if args.iter_override is not None and args.iter_override <= 0:
        parser.error("--iter-override must be a positive integer")
    if args.perf_record_frequency <= 0:
        parser.error("--perf-record-frequency must be a positive integer")
    if args.case_indices is not None:
        invalid_indices = [index for index in args.case_indices if index < 0]
        if invalid_indices:
            parser.error("--case-indices values must be non-negative integers")
        if args.run_mode != "case":
            parser.error("--case-indices requires --run-mode case")
    return args


def handle_interrupt(signum: int, frame: Any) -> None:
    INTERRUPT_EVENT.set()
    raise KeyboardInterrupt()


def print_line(line: str) -> None:
    with PRINT_LOCK:
        sys.stdout.write(line)
        sys.stdout.flush()


def register_active_process(prefix: str, process: subprocess.Popen[str]) -> None:
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES[process.pid] = (prefix, process)


def unregister_active_process(process: subprocess.Popen[str]) -> None:
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES.pop(process.pid, None)


def terminate_active_processes() -> None:
    INTERRUPT_EVENT.set()
    with ACTIVE_PROCESSES_LOCK:
        processes = list(ACTIVE_PROCESSES.values())

    for _, process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    for _, process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


atexit.register(terminate_active_processes)


def select_tests(config: RunConfig, args: argparse.Namespace) -> list[TestEntry]:
    if args.all:
        return config.tests

    requested = set(args.tests or [])
    selected: list[TestEntry] = []
    matched: set[str] = set()
    for test in config.tests:
        aliases = aliases_for_test(test)
        if requested & aliases:
            selected.append(test)
            matched.update(requested & aliases)

    missing = sorted(requested - matched)
    if missing:
        available = ", ".join(test.marker for test in config.tests)
        raise ValueError(
            f"requested test(s) not found in config: {', '.join(missing)}. "
            f"Available markers: {available}"
        )
    return selected


def aliases_for_test(test: TestEntry) -> set[str]:
    stem = Path(test.test_file).stem
    aliases = {test.marker, test.test_file, stem}
    if stem.startswith("test_"):
        aliases.add(stem[len("test_") :])
    return aliases


def build_run_dir(output_dir: Path, run_name: str | None) -> Path:
    base_name = run_name or datetime.now().strftime("profile_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = output_dir / base_name
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate

    suffix = 1
    while True:
        candidate = output_dir / f"{base_name}_{suffix}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        suffix += 1


def resolve_config_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.exists():
        return expanded
    if not expanded.is_absolute():
        script_relative = SCRIPT_DIR / expanded
        if script_relative.exists():
            return script_relative
        benchmark_relative = BENCHMARK_DIR / expanded
        if benchmark_relative.exists():
            return benchmark_relative
    return expanded


def resolve_counts(
    test: TestEntry, config: RunConfig, args: argparse.Namespace
) -> tuple[int, int]:
    tier = config.tiers[test.tier]
    warmup = tier.warmup
    if test.warmup_override is not None:
        warmup = test.warmup_override
    if args.warmup_override is not None:
        warmup = args.warmup_override

    iterations = tier.iterations
    if test.iter_override is not None:
        iterations = test.iter_override
    if args.iter_override is not None:
        iterations = args.iter_override
    return warmup, iterations


def resolve_dtypes(test: TestEntry, args: argparse.Namespace) -> list[str] | None:
    return args.dtypes if args.dtypes is not None else test.dtypes


def build_pytest_args(
    test: TestEntry,
    warmup: int,
    iterations: int,
    dtypes: list[str] | None,
    record: str,
    *,
    shape_file: str | Path | None = None,
    query: bool = False,
) -> list[str]:
    pytest_args = [
        test.test_file,
        "-s",
        "-m",
        test.marker,
        "--record",
        record,
        "--mode",
        test.mode,
        "--level",
        test.level,
        "--warmup",
        str(warmup),
        "--iter",
        str(iterations),
    ]
    metrics = test.metrics if test.metrics is not None else DEFAULT_METRICS
    if metrics:
        for metric in metrics:
            pytest_args.extend(["--metrics", metric])
    if dtypes:
        for dtype in dtypes:
            pytest_args.extend(["--dtypes", dtype])
    if shape_file is not None:
        pytest_args.extend(["--shape_file", str(shape_file)])
    if query:
        pytest_args.append("--query")
    return pytest_args


def expected_record_log_path(pytest_args: list[str]) -> Path:
    cmd_args = [
        arg.replace(".py", "").replace("=", "_").replace("/", "_")
        for arg in pytest_args
    ]
    log_file = "result_{}.log".format("_".join(cmd_args)).replace("_-", "-")
    return BENCHMARK_DIR / log_file


def build_environment(
    config: RunConfig,
    compile_log_path: Path | None,
    triton_cache_dir: Path | None,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(config.system.env_vars)

    python_path_entries = [str(PROFILING_DIR), str(FLAGGEMS_ROOT), str(SRC_DIR)]
    if env.get("PYTHONPATH"):
        python_path_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path_entries)

    if compile_log_path is not None:
        env["FLAGGEMS_COMPILE_LOG"] = str(compile_log_path)
    else:
        env.pop("FLAGGEMS_COMPILE_LOG", None)

    if triton_cache_dir is not None:
        env["TRITON_CACHE_DIR"] = str(triton_cache_dir)
    return env


def resolve_triton_cache_dir(
    config: RunConfig, op_dir: Path, jobs: int, *, isolated: bool = False
) -> Path | None:
    if isolated or jobs > 1:
        return op_dir / "triton_cache"
    configured = config.system.triton_cache_dir or os.environ.get("TRITON_CACHE_DIR")
    return Path(configured or "/tmp/triton_cache").expanduser()


def clear_triton_cache(cache_dir: Path | None) -> None:
    if cache_dir is None:
        return
    resolved = cache_dir.expanduser()
    if str(resolved) in {"", "/", "."}:
        raise ValueError(f"refusing to clear unsafe Triton cache path: {resolved}")
    shutil.rmtree(resolved, ignore_errors=True)


def apply_system_binding(command: list[str], config: RunConfig) -> list[str]:
    if config.system.cpu_affinity:
        command = ["taskset", "-c", config.system.cpu_affinity, *command]

    if config.system.numa_enabled:
        numa_command = ["numactl"]
        if config.system.numa_cpunodebind is not None:
            numa_command.append(f"--cpunodebind={config.system.numa_cpunodebind}")
        if config.system.numa_membind is not None:
            numa_command.append(f"--membind={config.system.numa_membind}")
        command = [*numa_command, *command]

    return command


def build_command(
    test: TestEntry,
    config: RunConfig,
    op_dir: Path,
    warmup: int,
    iterations: int,
    dtypes: list[str] | None,
    record: str,
    *,
    skip_perf: bool,
    jobs: int,
    perf_record: bool = False,
    perf_record_frequency: int = 99,
    perf_record_call_graph: str = "dwarf",
    shape_file: str | Path | None = None,
) -> tuple[
    list[str],
    Path | None,
    Path | None,
    Path | None,
    Path | None,
    PerfRecordRunner | None,
    list[str],
]:
    notes: list[str] = []
    pytest_args = build_pytest_args(
        test,
        warmup,
        iterations,
        dtypes,
        record,
        shape_file=shape_file,
    )
    target_cmd = [sys.executable, "-m", "pytest", *pytest_args]
    command = target_cmd
    perf_csv: Path | None = None
    perf_data_path: Path | None = None
    perf_report_path: Path | None = None
    perf_top_path: Path | None = None
    perf_record_runner: PerfRecordRunner | None = None

    if perf_record:
        if jobs > 1:
            notes.append("perf record disabled because --jobs > 1 would contend")
        elif PerfRecordRunner.is_available():
            perf_data_path = op_dir / "perf.data"
            perf_report_path = op_dir / "perf_report.txt"
            perf_top_path = op_dir / "perf_top10.txt"
            perf_record_runner = PerfRecordRunner(
                perf_data_path,
                perf_report_path,
                perf_top_path,
                frequency=perf_record_frequency,
                call_graph=perf_record_call_graph,
            )
            command = perf_record_runner.build_command(command)
            notes.append("perf stat skipped because --perf-record was enabled")
            skip_perf = True
        else:
            notes.append("perf not found in PATH; perf record skipped")

    perf_enabled = (
        config.profiling.cache_metrics_enabled
        and config.profiling.cache_backend == "perf"
    )
    if perf_record_runner is not None:
        pass
    elif perf_enabled and skip_perf:
        if jobs > 1:
            notes.append("perf disabled because --jobs > 1 would contend for counters")
        else:
            notes.append("perf skipped by --skip-perf")
    elif perf_enabled:
        if PerfStatRunner.is_available():
            perf_csv = op_dir / "perf_stat.csv"
            command = PerfStatRunner(
                config.profiling.cache_events, perf_csv
            ).build_command(command)
        else:
            notes.append("perf not found in PATH; cache metrics skipped")
    elif config.profiling.cache_metrics_enabled and config.profiling.cache_backend != "perf":
        notes.append(
            f"unsupported cache metrics backend {config.profiling.cache_backend!r}; skipped"
        )

    return (
        apply_system_binding(command, config),
        perf_csv,
        perf_data_path,
        perf_report_path,
        perf_top_path,
        perf_record_runner,
        notes,
    )


def build_query_command(
    test: TestEntry,
    config: RunConfig,
    warmup: int,
    iterations: int,
) -> tuple[list[str], list[str], Path]:
    pytest_args = build_pytest_args(
        test,
        warmup,
        iterations,
        None,
        "log",
        query=True,
    )
    command = apply_system_binding(
        [sys.executable, "-m", "pytest", *pytest_args], config
    )
    expected_log = expected_record_log_path(pytest_args)
    return command, pytest_args, expected_log


def stream_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    prefix: str,
    cpu_memory_enabled: bool,
    cpu_memory_interval_sec: float,
    cpu_memory_csv: Path | None,
) -> tuple[int, str, float, str | None]:
    if INTERRUPT_EVENT.is_set():
        return -signal.SIGTERM, "", 0.0, "cancelled by interrupt"

    start = time.perf_counter()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        return 127, f"{exc}\n", time.perf_counter() - start, str(exc)

    register_active_process(prefix, process)
    monitor = None
    monitor_note = None
    if cpu_memory_enabled and cpu_memory_csv is not None:
        try:
            monitor = ResourceMonitor(process.pid, cpu_memory_interval_sec)
            monitor.start()
        except Exception as exc:
            monitor_note = f"CPU/memory profiling skipped: {exc}"

    output_lines: list[str] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print_line(f"[{prefix}] {line}")
            output_lines.append(line)
        returncode = process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if monitor is not None:
            monitor.stop()
            monitor.dump_csv(cpu_memory_csv)
        unregister_active_process(process)

    return returncode, "".join(output_lines), time.perf_counter() - start, monitor_note


def run_single_op(
    test: TestEntry,
    config: RunConfig,
    run_dir: Path,
    args: argparse.Namespace,
    *,
    jobs: int,
    skip_perf: bool,
    op_dir: Path | None = None,
    run_mode: str = "group",
    run_label: str | None = None,
    case_index: int | None = None,
    requested_shape: Any = None,
    shape_file: str | Path | None = None,
    isolated_cache: bool = False,
) -> RunArtifacts:
    op_name = test.marker
    op_dir = op_dir or run_dir / safe_name(op_name)
    op_dir.mkdir(parents=True, exist_ok=True)
    label = run_label or op_name

    warmup, iterations = resolve_counts(test, config, args)
    dtypes = resolve_dtypes(test, args)
    compile_log_path = (
        op_dir / "compile.jsonl" if config.profiling.compilation_time_enabled else None
    )
    cpu_memory_csv = (
        op_dir / "cpu_memory.csv"
        if config.profiling.cpu_memory_enabled and not args.skip_cpu_mem
        else None
    )
    triton_cache_dir = resolve_triton_cache_dir(
        config, op_dir, jobs, isolated=isolated_cache
    )
    (
        command,
        perf_csv,
        perf_data_path,
        perf_report_path,
        perf_top_path,
        perf_record_runner,
        command_notes,
    ) = build_command(
        test,
        config,
        op_dir,
        warmup,
        iterations,
        dtypes,
        args.record,
        skip_perf=skip_perf,
        jobs=jobs,
        perf_record=args.perf_record,
        perf_record_frequency=args.perf_record_frequency,
        perf_record_call_graph=args.perf_record_call_graph,
        shape_file=shape_file,
    )
    (op_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")

    pytest_args = build_pytest_args(
        test,
        warmup,
        iterations,
        dtypes,
        args.record,
        shape_file=shape_file,
    )
    expected_log = (
        expected_record_log_path(pytest_args) if args.record == "log" else None
    )
    if expected_log is not None:
        expected_log.unlink(missing_ok=True)

    env = build_environment(config, compile_log_path, triton_cache_dir)
    stdout_path = op_dir / "stdout.txt"
    notes = command_notes[:]

    try:
        if config.system.clear_triton_cache:
            clear_triton_cache(triton_cache_dir)
    except Exception as exc:
        note = f"failed to clear Triton cache: {exc}"
        stdout_path.write_text(note + "\n", encoding="utf-8")
        return RunArtifacts(
            op_name=op_name,
            test_file=test.test_file,
            marker=test.marker,
            status="failed",
            returncode=1,
            note=note,
            wall_time_sec=0.0,
            compile_log_path=compile_log_path,
            cpu_memory_csv=cpu_memory_csv,
            perf_csv=perf_csv,
            perf_data_path=perf_data_path,
            perf_report_path=perf_report_path,
            perf_top_path=perf_top_path,
            record_log_path=None,
            stdout_path=stdout_path,
            command=command,
            warmup=warmup,
            iterations=iterations,
            mode=test.mode,
            level=test.level,
            dtypes=dtypes,
            record=args.record,
            run_mode=run_mode,
            run_label=label,
            case_index=case_index,
            requested_shape=requested_shape,
        )

    print_line(f"\n[profile] starting {label}: {shlex.join(command)}\n")
    returncode, raw_output, wall_time_sec, monitor_note = stream_subprocess(
        command,
        cwd=BENCHMARK_DIR,
        env=env,
        prefix=label,
        cpu_memory_enabled=cpu_memory_csv is not None,
        cpu_memory_interval_sec=config.profiling.cpu_memory_interval_sec,
        cpu_memory_csv=cpu_memory_csv,
    )
    if monitor_note:
        notes.append(monitor_note)
    if perf_record_runner is not None:
        perf_record_result = perf_record_runner.write_reports()
        if perf_record_result.note:
            notes.append(perf_record_result.note)

    stdout_path.write_text(raw_output, encoding="utf-8")

    record_log_path = None
    benchmark_results: list[dict[str, Any]] = []
    if expected_log is not None and expected_log.exists():
        record_log_path = op_dir / "record.log"
        shutil.move(str(expected_log), record_log_path)
        benchmark_results = parse_record_log(record_log_path)

    status, status_note = determine_status(returncode, benchmark_results, raw_output)
    if status_note:
        notes.append(status_note)
    note = "; ".join(notes) if notes else None

    print_line(
        f"[profile] completed {label}: status={status}"
        + (f", note={note}" if note else "")
        + "\n"
    )
    return RunArtifacts(
        op_name=op_name,
        test_file=test.test_file,
        marker=test.marker,
        status=status,
        returncode=returncode,
        note=note,
        wall_time_sec=wall_time_sec,
        compile_log_path=compile_log_path,
        cpu_memory_csv=cpu_memory_csv,
        perf_csv=perf_csv,
        perf_data_path=perf_data_path,
        perf_report_path=perf_report_path,
        perf_top_path=perf_top_path,
        record_log_path=record_log_path,
        stdout_path=stdout_path,
        command=command,
        warmup=warmup,
        iterations=iterations,
        mode=test.mode,
        level=test.level,
        dtypes=dtypes,
        record=args.record,
        run_mode=run_mode,
        run_label=label,
        case_index=case_index,
        requested_shape=requested_shape,
    )


def discover_case_specs(
    test: TestEntry,
    config: RunConfig,
    run_dir: Path,
    args: argparse.Namespace,
) -> list[CaseSpec]:
    op_root = run_dir / safe_name(test.marker)
    query_dir = op_root / "_query"
    query_dir.mkdir(parents=True, exist_ok=True)
    warmup, iterations = resolve_counts(test, config, args)
    command, pytest_args, expected_log = build_query_command(
        test, config, warmup, iterations
    )
    (query_dir / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    expected_log.unlink(missing_ok=True)

    env = build_environment(config, None, None)
    stdout_path = query_dir / "stdout.txt"
    print_line(
        f"\n[profile] discovering shapes for {test.marker}: "
        f"{shlex.join(command)}\n"
    )
    returncode, raw_output, _, _ = stream_subprocess(
        command,
        cwd=BENCHMARK_DIR,
        env=env,
        prefix=f"{test.marker}:query",
        cpu_memory_enabled=False,
        cpu_memory_interval_sec=config.profiling.cpu_memory_interval_sec,
        cpu_memory_csv=None,
    )
    stdout_path.write_text(raw_output, encoding="utf-8")

    query_log = None
    if expected_log.exists():
        query_log = query_dir / "record.log"
        shutil.move(str(expected_log), query_log)

    if returncode != 0:
        raise RuntimeError(
            f"shape query for {test.marker} failed with pytest exit code {returncode}"
        )
    if query_log is None:
        raise RuntimeError(f"shape query for {test.marker} produced no record log")

    shapes, shape_desc = parse_query_shapes(query_log, test.marker)
    if not shapes:
        raise RuntimeError(f"shape query for {test.marker} produced no shapes")

    case_specs = []
    selected_indices = set(args.case_indices or [])
    for index, shape in enumerate(shapes):
        if args.case_indices is not None and index not in selected_indices:
            continue
        case_dir = op_root / f"case_{index:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        shape_file = case_dir / "shape.yaml"
        write_case_shape_file(shape_file, test.marker, shape, shape_desc)
        case_specs.append(
            CaseSpec(
                index=index,
                shape=shape,
                shape_desc=shape_desc,
                shape_file=shape_file,
                shape_file_arg=benchmark_relative_path(shape_file),
            )
        )
    if not case_specs:
        selected = ", ".join(str(index) for index in args.case_indices or [])
        raise RuntimeError(
            f"shape query for {test.marker} found {len(shapes)} shapes, "
            f"but --case-indices selected no runnable cases: {selected}"
        )
    return case_specs


def run_case_mode_for_op(
    test: TestEntry,
    config: RunConfig,
    run_dir: Path,
    args: argparse.Namespace,
    *,
    jobs: int,
    skip_perf: bool,
) -> list[RunArtifacts]:
    try:
        case_specs = discover_case_specs(test, config, run_dir, args)
    except Exception as exc:
        return [
            build_failed_artifact(
                test,
                config,
                run_dir,
                args,
                note=f"shape query failed: {exc}",
                run_label=f"{test.marker}/_query",
                run_mode="case",
            )
        ]

    def run_case(spec: CaseSpec) -> RunArtifacts:
        label = f"{test.marker}/case_{spec.index:03d}"
        return run_single_op(
            test,
            config,
            run_dir,
            args,
            jobs=jobs,
            skip_perf=skip_perf,
            op_dir=run_dir / safe_name(test.marker) / f"case_{spec.index:03d}",
            run_mode="case",
            run_label=label,
            case_index=spec.index,
            requested_shape=spec.shape,
            shape_file=spec.shape_file_arg,
            isolated_cache=True,
        )

    if jobs == 1:
        return [run_case(spec) for spec in case_specs]

    artifacts: list[RunArtifacts] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(run_case, spec): spec for spec in case_specs}
        for future in as_completed(futures):
            artifacts.append(future.result())
    return sorted(artifacts, key=lambda item: item.case_index or 0)


def build_failed_artifact(
    test: TestEntry,
    config: RunConfig,
    run_dir: Path,
    args: argparse.Namespace,
    *,
    note: str,
    run_label: str,
    run_mode: str,
) -> RunArtifacts:
    warmup, iterations = resolve_counts(test, config, args)
    dtypes = resolve_dtypes(test, args)
    op_dir = run_dir / safe_name(test.marker) / "_query"
    op_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = op_dir / "stdout.txt"
    if not stdout_path.exists():
        stdout_path.write_text(note + "\n", encoding="utf-8")
    return RunArtifacts(
        op_name=test.marker,
        test_file=test.test_file,
        marker=test.marker,
        status="failed",
        returncode=1,
        note=note,
        wall_time_sec=0.0,
        compile_log_path=None,
        cpu_memory_csv=None,
        perf_csv=None,
        perf_data_path=None,
        perf_report_path=None,
        perf_top_path=None,
        record_log_path=None,
        stdout_path=stdout_path,
        command=[],
        warmup=warmup,
        iterations=iterations,
        mode=test.mode,
        level=test.level,
        dtypes=dtypes,
        record=args.record,
        run_mode=run_mode,
        run_label=run_label,
    )


def parse_query_shapes(
    record_log_path: Path,
    marker: str,
) -> tuple[list[Any], str | None]:
    shapes: list[Any] = []
    shape_desc: str | None = None
    for line in record_log_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if not line.startswith("[INFO] "):
            continue
        body = line[len("[INFO] ") :].strip()
        if not body.startswith("{"):
            continue
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            try:
                payload = ast.literal_eval(body)
            except (SyntaxError, ValueError):
                continue
        if not isinstance(payload, dict):
            continue
        if payload.get("op_name") != marker:
            continue
        recommended = payload.get("recommended_core_shapes")
        if isinstance(recommended, list):
            shapes = [_normalize_shape(shape) for shape in recommended]
            shape_desc = (
                str(payload.get("shape_desc")) if payload.get("shape_desc") else None
            )
    return shapes, shape_desc


def write_case_shape_file(
    path: Path,
    op_name: str,
    shape: Any,
    shape_desc: str | None,
) -> None:
    lines = [
        f"{op_name}:",
        "  shapes:",
        f"    - {json.dumps(_normalize_shape(shape), ensure_ascii=False)}",
    ]
    if shape_desc:
        lines.append(f"  shape_desc: {json.dumps(shape_desc, ensure_ascii=False)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark_relative_path(path: Path) -> Path:
    try:
        return path.relative_to(BENCHMARK_DIR)
    except ValueError:
        return path


def _normalize_shape(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize_shape(item) for item in value]
    if isinstance(value, list):
        return [_normalize_shape(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_shape(item) for key, item in value.items()}
    return value


def print_dry_run(
    tests: list[TestEntry],
    config: RunConfig,
    args: argparse.Namespace,
    *,
    jobs: int,
    skip_perf: bool,
) -> None:
    run_name = args.run_name or datetime.now().strftime("profile_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    for test in tests:
        op_dir = run_dir / safe_name(test.marker)
        warmup, iterations = resolve_counts(test, config, args)
        dtypes = resolve_dtypes(test, args)
        command, _, _, _, _, _, notes = build_command(
            test,
            config,
            op_dir,
            warmup,
            iterations,
            dtypes,
            args.record,
            skip_perf=skip_perf,
            jobs=jobs,
            perf_record=args.perf_record,
            perf_record_frequency=args.perf_record_frequency,
            perf_record_call_graph=args.perf_record_call_graph,
        )
        compile_log = op_dir / "compile.jsonl"
        cache_dir = resolve_triton_cache_dir(config, op_dir, jobs)
        print(f"[dry-run] {test.marker}")
        print(f"  cwd: {BENCHMARK_DIR}")
        print(f"  TRITON_CACHE_DIR: {cache_dir}")
        if config.profiling.compilation_time_enabled:
            print(f"  FLAGGEMS_COMPILE_LOG: {compile_log}")
        else:
            print("  FLAGGEMS_COMPILE_LOG: disabled")
        if notes:
            print(f"  notes: {'; '.join(notes)}")
        if args.run_mode == "case":
            query_command, _, _ = build_query_command(test, config, warmup, iterations)
            print("  case mode query cmd:")
            print(f"    {shlex.join(query_command)}")
            if args.case_indices is not None:
                selected = ", ".join(str(index) for index in args.case_indices)
                print(f"  selected case indices: {selected}")
            print("  case commands are generated after the query returns shapes.")
        else:
            print(f"  cmd: {shlex.join(command)}")


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def aggregate_results(
    run_dir: Path,
    generated_at: str,
    config: RunConfig,
    system_info: dict[str, Any],
    artifacts: list[RunArtifacts],
    run_mode: str,
) -> None:
    aggregator = ResultAggregator(
        run_dir,
        generated_at=generated_at,
        config_file=config.config_path,
        system_info=system_info,
        run_mode=run_mode,
    )
    for item in artifacts:
        aggregator.add_op_result(
            item.op_name,
            test_file=item.test_file,
            marker=item.marker,
            status=item.status,
            returncode=item.returncode,
            wall_time_sec=item.wall_time_sec,
            compile_log_path=item.compile_log_path,
            cpu_memory_csv=item.cpu_memory_csv,
            perf_csv=item.perf_csv,
            perf_data_path=item.perf_data_path,
            perf_report_path=item.perf_report_path,
            perf_top_path=item.perf_top_path,
            record_log_path=item.record_log_path,
            stdout_path=item.stdout_path,
            command=item.command,
            note=item.note,
            warmup=item.warmup,
            iterations=item.iterations,
            mode=item.mode,
            level=item.level,
            dtypes=item.dtypes,
            record=item.record,
            run_mode=item.run_mode,
            run_label=item.run_label,
            case_index=item.case_index,
            requested_shape=item.requested_shape,
        )
    aggregator.write_summary(run_dir)


def main() -> int:
    args = parse_args()
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_interrupt)

    try:
        config = RunConfig.from_yaml(
            resolve_config_path(args.config), benchmark_dir=BENCHMARK_DIR
        )
        selected_tests = select_tests(config, args)
    except Exception as exc:
        print(f"[profile] {exc}", file=sys.stderr)
        return 2

    if args.run_mode == "case":
        jobs = max(1, args.jobs)
    else:
        jobs = max(1, min(args.jobs, len(selected_tests)))
    skip_perf = args.skip_perf or jobs > 1
    if jobs > 1 and not args.skip_perf:
        print_line("[profile] --jobs > 1 detected; perf collection is disabled.\n")

    if args.dry_run:
        print_dry_run(selected_tests, config, args, jobs=jobs, skip_perf=skip_perf)
        return 0

    run_dir = build_run_dir(args.output_dir, args.run_name)
    generated_at = datetime.now().isoformat(timespec="seconds")
    shutil.copy2(config.config_path, run_dir / "config_snapshot.yaml")
    system_info = write_system_info(run_dir)

    artifacts: list[RunArtifacts] = []
    try:
        if args.run_mode == "case":
            for test in selected_tests:
                artifacts.extend(
                    run_case_mode_for_op(
                        test,
                        config,
                        run_dir,
                        args,
                        jobs=jobs,
                        skip_perf=skip_perf,
                    )
                )
        elif jobs == 1:
            for test in selected_tests:
                artifacts.append(
                    run_single_op(
                        test,
                        config,
                        run_dir,
                        args,
                        jobs=jobs,
                        skip_perf=skip_perf,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(
                        run_single_op,
                        test,
                        config,
                        run_dir,
                        args,
                        jobs=jobs,
                        skip_perf=skip_perf,
                    ): test.marker
                    for test in selected_tests
                }
                for future in as_completed(futures):
                    artifacts.append(future.result())
    except KeyboardInterrupt:
        terminate_active_processes()
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)

    aggregate_results(
        run_dir, generated_at, config, system_info, artifacts, args.run_mode
    )
    print_line(f"[profile] wrote summary to {run_dir / 'summary.json'}\n")
    print_line(f"[profile] wrote markdown to {run_dir / 'summary.md'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
