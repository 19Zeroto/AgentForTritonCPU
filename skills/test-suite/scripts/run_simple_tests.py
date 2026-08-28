#!/usr/bin/env python3
"""
Simple Test Runner

Runs test files in parallel with per-file cache clearing.

Features:
  - N concurrent file processes, M xdist threads per file
  - Clears $TRITON_CACHE_DIR after each file completes
  - Persists last 500 lines of log for failed files by default
  - Can persist full pytest output for every file
  - Simple state file for resume support

Usage:
  python3 scripts/run_simple_tests.py
  python3 scripts/run_simple_tests.py -N 4 -M 4
  python3 scripts/run_simple_tests.py -i test_unary -e test_DSA -r
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
AGENTFORTRITONCPU_DIR = SCRIPT_DIR.parent.parent.parent
AGENT_DIR = Path(
    os.environ.get("AGENT_DIR", AGENTFORTRITONCPU_DIR.parent)
).expanduser().resolve()
TRITON_REPO_DIR = Path(
    os.environ.get("TRITON_REPO_DIR", AGENT_DIR / "triton-cpu")
).expanduser().resolve()
OUTPUT_DIR = Path(
    os.environ.get(
        "AGENTFORTRITONCPU_TEST_OUTPUT_DIR",
        AGENT_DIR / "logs" / "AgentForTritonCPU" / "test-suite",
    )
).expanduser().resolve()
LOG_DIR = str(OUTPUT_DIR / "file-logs")
DEFAULT_STATE_FILE = str(OUTPUT_DIR / "simple-test-state.json")
DEFAULT_TESTS_DIR = str(TRITON_REPO_DIR / "FlagGems" / "tests")
TAIL_LINES = 500

_LOCK = threading.Lock()


def log(msg: str):
    with _LOCK:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)


def save_pytest_log(rel_file: str, output: str, log_dir: str,
                    *, tail_only: bool, header: str = ""):
    logpath = Path(log_dir) / Path(rel_file).with_suffix(".log")
    logpath.parent.mkdir(parents=True, exist_ok=True)
    body = output
    if tail_only:
        lines = output.splitlines()
        body = "\n".join(lines[-TAIL_LINES:] if len(lines) > TAIL_LINES else lines)
    if header:
        body = header + body
    logpath.write_text(body, encoding="utf-8")
    return str(logpath)


def clear_triton_cache():
    cache_dir = os.environ.get("TRITON_CACHE_DIR", "")
    if not cache_dir:
        return
    cache_path = Path(cache_dir)
    if not cache_path.is_dir():
        return
    shutil.rmtree(cache_path, ignore_errors=True)
    cache_path.mkdir(parents=True, exist_ok=True)
    log(f"CACHE cleared {cache_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Simple Test Runner")
    parser.add_argument("-d", "--tests-dir", default=DEFAULT_TESTS_DIR,
                        help=f"Test directory (default: {DEFAULT_TESTS_DIR})")
    parser.add_argument("-N", "--files", type=int, default=4,
                        help="Max concurrent file processes (default: 4)")
    parser.add_argument("-M", "--threads", type=int, default=4,
                        help="pytest-xdist threads per file (default: 4)")
    parser.add_argument("-i", "--include",
                        help="Regex pattern for file path inclusion")
    parser.add_argument("-e", "--exclude",
                        help="Regex pattern for file path exclusion")
    parser.add_argument("-r", "--resume", action="store_true",
                        help="Resume: skip already-completed files")
    parser.add_argument("-s", "--state-file",
                        default=DEFAULT_STATE_FILE,
                        help=f"State file (default: {DEFAULT_STATE_FILE})")
    parser.add_argument("--log-dir", default=LOG_DIR,
                        help=f"Directory for pytest logs (default: {LOG_DIR})")
    parser.add_argument("--save-all-logs", action="store_true",
                        help="Persist full pytest output for every test file")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_test_files(tests_dir: str, include: Optional[str],
                        exclude: Optional[str]):
    tests_path = Path(tests_dir).resolve()
    if not tests_path.is_dir():
        print(f"Error: test directory not found: {tests_dir}", file=sys.stderr)
        sys.exit(1)

    all_files = sorted(tests_path.rglob("test_*.py"))
    inc_re = re.compile(include) if include else None
    exc_re = re.compile(exclude) if exclude else None

    result = []
    for f in all_files:
        if inc_re and not inc_re.search(str(f)):
            continue
        if exc_re and exc_re.search(str(f)):
            continue
        result.append(str(f))
    return result


def rel_path(file_path: str, tests_dir: str) -> str:
    return str(Path(file_path).resolve().relative_to(Path(tests_dir).resolve()))


# ---------------------------------------------------------------------------
# State persistence (v1: file-level flat dict)
# ---------------------------------------------------------------------------

class StateManager:
    def __init__(self, state_file: str):
        self._path = Path(state_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.state = self._load()

    def _load(self):
        if not self._path.exists():
            return {"version": 1, "results": {}}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if data.get("version") == 1:
                return data
            return {"version": 1, "results": {}}
        except (json.JSONDecodeError, OSError):
            return {"version": 1, "results": {}}

    def save(self):
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with self._lock:
            tmp.write_text(json.dumps(self.state, indent=2,
                                       ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._path)

    def get(self, key: str):
        return self.state["results"].get(key)

    def upsert(self, key: str, entry: dict):
        with self._lock:
            self.state["results"][key] = entry

    def is_done(self, key: str) -> bool:
        entry = self.get(key)
        return entry is not None and entry.get("status") == "done"

    def summary_totals(self):
        passed = sum(v.get("passed", 0) for v in self.state["results"].values())
        failed = sum(v.get("failed", 0) for v in self.state["results"].values())
        skipped = sum(v.get("skipped", 0) for v in self.state["results"].values())
        error = sum(v.get("error", 0) for v in self.state["results"].values())
        duration = sum(v.get("duration", 0.0) for v in self.state["results"].values())
        return {"passed": passed, "failed": failed, "skipped": skipped,
                "error": error, "duration": duration}


# ---------------------------------------------------------------------------
# Pytest execution
# ---------------------------------------------------------------------------

_SUMMARY_RE = re.compile(r"(\d+)\s+(passed|failed|skipped|error)", re.IGNORECASE)


def parse_pytest_summary(text: str):
    result = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for m in _SUMMARY_RE.finditer(text):
        result[m.group(2).lower()] = int(m.group(1))
    return result


def format_result_line(r: dict):
    parts = []
    if r["passed"]:
        parts.append(f"{r['passed']} passed")
    if r["failed"]:
        parts.append(f"{r['failed']} failed")
    if r["skipped"]:
        parts.append(f"{r['skipped']} skipped")
    if r["error"]:
        parts.append(f"{r['error']} error")
    return ", ".join(parts) if parts else "0 tests"


def run_file(abs_file: str, rel_file: str, threads: int,
             state_mgr: StateManager, log_dir: str, save_all_logs: bool):
    key = rel_file
    fname = Path(abs_file).name

    log(f"START {fname}")

    cmd = ["pytest", abs_file, "--tb=short", "-q", "-p", "no:cacheprovider"]
    if threads > 0:
        cmd.extend(["-n", str(threads)])
    command_str = shlex.join(cmd)
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    env["COVERAGE_PROCESS_START"] = ""

    r = subprocess.run(cmd, capture_output=True, text=True, env=env)

    duration = time.time() - t0
    summary = parse_pytest_summary(r.stdout + r.stderr)
    if r.returncode != 0 and summary["failed"] == 0 and summary["error"] == 0:
        summary["error"] = 1
    summary["duration"] = duration
    summary["exit_code"] = r.returncode
    summary["command"] = command_str

    has_fail = summary["failed"] > 0 or summary["error"] > 0

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_output = r.stdout + r.stderr
    log_header = (
        f"command: {command_str}\n"
        f"file: {rel_file}\n"
        f"started_at: {started_at}\n"
        f"finished_at: {ts}\n"
        f"exit_code: {r.returncode}\n\n"
    )

    if has_fail:
        logpath = save_pytest_log(
            rel_file, full_output, log_dir,
            tail_only=not save_all_logs, header=log_header,
        )
        summary["log"] = logpath
        log(f"FAIL  {fname}  {format_result_line(summary)}  {duration:.1f}s  log: {logpath}")
    else:
        if save_all_logs:
            logpath = save_pytest_log(
                rel_file, full_output, log_dir,
                tail_only=False, header=log_header,
            )
            summary["log"] = logpath
            log(f"PASS  {fname}  {format_result_line(summary)}  {duration:.1f}s  log: {logpath}")
        else:
            summary["log"] = ""
            log(f"PASS  {fname}  {format_result_line(summary)}  {duration:.1f}s")

    clear_triton_cache()

    entry = {
        "file": rel_file,
        "passed": summary["passed"],
        "failed": summary["failed"],
        "skipped": summary["skipped"],
        "error": summary["error"],
        "duration": summary["duration"],
        "log": summary.get("log", ""),
        "command": summary.get("command", ""),
        "exit_code": summary.get("exit_code", 0),
        "timestamp": ts,
        "status": "done",
    }

    state_mgr.upsert(key, entry)
    state_mgr.save()
    return entry


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    state_mgr = StateManager(args.state_file)

    print("-- Discovery --")

    files = discover_test_files(args.tests_dir, args.include, args.exclude)
    if not files:
        print("No test files found. Aborting.")
        sys.exit(1)
    print(f"  Files: {len(files)}")

    exec_files = []
    for f in files:
        rf = rel_path(f, args.tests_dir)
        if args.resume and state_mgr.is_done(rf):
            continue
        exec_files.append((f, rf))

    if args.resume:
        skipped = len(files) - len(exec_files)
        print(f"  Resume: {len(exec_files)} to run, {skipped} skipped")
    else:
        print(f"  Execution: {len(exec_files)} files, "
              f"{args.files} concurrent, {args.threads} threads each")

    if not exec_files:
        print("Nothing to do.")
        return

    total = len(exec_files)
    done_count = 0
    dc_lock = threading.Lock()

    def on_done(future):
        nonlocal done_count
        try:
            future.result()
        except Exception as e:
            log(f"ERROR: {e}")
        with dc_lock:
            done_count += 1
            cur = done_count
        s = state_mgr.summary_totals()
        print(f"  [{cur:3d}/{total:3d}]  "
              f"passed={s['passed']}  failed={s['failed']}  "
              f"skipped={s['skipped']}  |  {s['duration']:.1f}s", flush=True)

    pool = ThreadPoolExecutor(max_workers=args.files)
    futures = []
    for f, rf in exec_files:
        fut = pool.submit(
            run_file, f, rf, args.threads, state_mgr,
            args.log_dir, args.save_all_logs,
        )
        futures.append(fut)

    for fut in as_completed(futures):
        on_done(fut)

    pool.shutdown(wait=False)

    s = state_mgr.summary_totals()
    print()
    print("=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Files:        {total}")
    print(f"  Tests passed: {s['passed']}")
    print(f"  Tests failed: {s['failed']}")
    print(f"  Tests skipped:{s['skipped']}")
    print(f"  Tests error:  {s['error']}")
    print(f"  Total time:   {s['duration']:.1f}s")
    print("=" * 60)

    failed_entries = [v for v in state_mgr.state["results"].values()
                     if v.get("failed", 0) > 0 or v.get("error", 0) > 0]
    if failed_entries:
        print("\n  Failed files:")
        for v in failed_entries:
            print(f"    {v['file']}  "
                  f"(passed={v['passed']} failed={v['failed']} "
                  f"skipped={v.get('skipped',0)})  "
                  f"duration={v.get('duration',0.0):.1f}s")
            if v.get("log"):
                print(f"      log: {v['log']}")

    if s["failed"] > 0 or s["error"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
