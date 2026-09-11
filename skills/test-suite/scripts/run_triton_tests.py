#!/usr/bin/env python3
"""
Auto Triton Test Runner

Multi-threaded pytest runner with:
  - (file, marker) granularity, each pair runs as one subprocess
  - Timestamped log output (start / done+summary per work item)
  - Full pytest log persistence once per failed file
  - State persistence (v3: remark-level flat dict, additive across runs)
  - --resume skips completed items; re-running same marker updates in-place
  - -f/--force clears cached state before running
  - --clear-cache clears cached state and exits
  - Auto marker discovery from test files
  - Include/exclude regex file filtering plus global file exclusions

Usage:
  python3 scripts/run_triton_tests.py
  python3 scripts/run_triton_tests.py -m abs,cos
  python3 scripts/run_triton_tests.py -i test_unary -e test_DSA
  python3 scripts/run_triton_tests.py -f
  python3 scripts/run_triton_tests.py --clear-cache
  python3 scripts/run_triton_tests.py -w 8 -r
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUILTIN_MARKERS = frozenset({
    "parametrize", "skip", "skipif", "xfail", "usefixtures",
    "filterwarnings", "timeout", "tryfirst", "trylast",
})

SCRIPT_DIR = Path(__file__).resolve().parent
AGENTFORTRITONCPU_DIR = SCRIPT_DIR.parent.parent.parent
AGENT_DIR = Path(
    os.environ.get("AGENT_DIR", Path.home() / "agent")
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
LOG_DIR = str(OUTPUT_DIR / "marker-logs")
DEFAULT_STATE_FILE = str(OUTPUT_DIR / "triton-test-state.json")
DEFAULT_TESTS_DIR = str(TRITON_REPO_DIR / "FlagGems" / "tests")

# Add relative paths or basenames here to exclude all markers in those files.
# Examples:
#   "test_DSA/test_bin_topk.py"
#   "test_bin_topk.py"
GLOBAL_EXCLUDED_TEST_FILES = [
]

# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()

def log(msg: str):
    with _LOCK:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)


def _safe_log_stem(rel_file: str) -> str:
    rel = _normalize_file_token(rel_file)
    return Path(rel).with_suffix("").as_posix().replace("/", "__")


def save_file_log(rel_file: str, failed_outputs):
    logname = f"{_safe_log_stem(rel_file)}.log"
    logpath = Path(LOG_DIR) / logname
    logpath.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for item in failed_outputs:
        entry = item["entry"]
        chunks.append(
            "\n".join([
                "=" * 80,
                f"file: {entry['file']}",
                f"marker: {entry['marker']}",
                f"exit_code: {entry.get('exit_code', 0)}",
                f"result: {format_result_line(entry)}",
                f"duration: {entry.get('duration', 0.0):.1f}s",
                "-" * 80,
                item["output"].rstrip(),
                "",
            ])
        )
    logpath.write_text("\n".join(chunks), encoding="utf-8")
    return str(logpath)


def _state_tmp_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def clear_script_cache(state_file: str):
    state_path = Path(state_file)
    cleared = []
    for path in (state_path, _state_tmp_path(state_path)):
        try:
            path.unlink()
            cleared.append(str(path))
        except FileNotFoundError:
            pass
    return cleared

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Auto Triton Test Runner")
    parser.add_argument("-d", "--tests-dir", default=DEFAULT_TESTS_DIR,
                        help=f"Test directory (default: {DEFAULT_TESTS_DIR})")
    parser.add_argument("-m", "--markers",
                        help="Comma-separated markers to run; auto-discover if omitted")
    parser.add_argument("-i", "--include",
                        help="Regex pattern for file path inclusion")
    parser.add_argument("-e", "--exclude",
                        help="Regex pattern for file path exclusion")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="Max parallel pytest processes (default: 4)")
    parser.add_argument("-s", "--state-file", default=DEFAULT_STATE_FILE,
                        help=f"State persistence file (default: {DEFAULT_STATE_FILE})")
    parser.add_argument("-x", "--pytest-xdist", type=int, default=4, metavar="N",
                        help="Add -n N for pytest-xdist parallelism inside each work item (default: 4)")
    parser.add_argument("-r", "--resume", action="store_true",
                        help="Resume from last interrupted run")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Clear cached state before running; recount and run all discovered items")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear cached state file and exit")
    return parser.parse_args(argv)

# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

MARKER_RE = re.compile(r"@pytest\.mark\.(\w+)")
NODEID_RE = re.compile(r"::")


def _normalize_file_token(path: str) -> str:
    token = str(path).strip().replace("\\", "/")
    if token.startswith("./"):
        token = token[2:]
    return token


def _global_exclude_sets():
    rel_names = set()
    abs_names = set()
    for item in GLOBAL_EXCLUDED_TEST_FILES:
        token = _normalize_file_token(item)
        if not token:
            continue
        p = Path(token)
        if p.is_absolute():
            abs_names.add(str(p.resolve()))
        else:
            rel_names.add(token)
    return rel_names, abs_names


def discover_test_files(tests_dir: str, include: Optional[str], exclude: Optional[str]):
    tests_path = Path(tests_dir).resolve()
    if not tests_path.is_dir():
        print(f"Error: test directory not found: {tests_dir}", file=sys.stderr)
        sys.exit(1)

    all_files = sorted(tests_path.rglob("test_*.py"))
    inc_re = re.compile(include) if include else None
    exc_re = re.compile(exclude) if exclude else None
    global_rel_excludes, global_abs_excludes = _global_exclude_sets()

    result = []
    global_excluded = []
    for f in all_files:
        rel = _normalize_file_token(str(f.resolve().relative_to(tests_path)))
        if inc_re and not inc_re.search(str(f)):
            continue
        if exc_re and exc_re.search(str(f)):
            continue
        if (rel in global_rel_excludes or f.name in global_rel_excludes or
                str(f.resolve()) in global_abs_excludes):
            global_excluded.append(str(f))
            continue
        result.append(str(f))
    return result, global_excluded


def rel_path(file_path: str, tests_dir: str) -> str:
    return str(Path(file_path).resolve().relative_to(Path(tests_dir).resolve()))


def extract_markers(file_path: str):
    try:
        text = Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return set()
    markers = set()
    for m in MARKER_RE.finditer(text):
        name = m.group(1)
        if name not in BUILTIN_MARKERS:
            markers.add(name)
    return markers


_COLLECT_ENV = None

def _collect_env():
    global _COLLECT_ENV
    if _COLLECT_ENV is None:
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        env["COVERAGE_PROCESS_START"] = ""
        _COLLECT_ENV = env
    return _COLLECT_ENV


def count_tests_for_item(file_path: str, marker: str):
    cmd = [
        "pytest", file_path,
        "-m", marker,
        "--collect-only", "-q",
        "-p", "no:cacheprovider",
    ]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=_collect_env(),
            cwd=TRITON_REPO_DIR,
        )
        count = 0
        for line in r.stdout.splitlines():
            if not line.strip():
                continue
            if NODEID_RE.search(line.strip()):
                count += 1
        return count
    except subprocess.TimeoutExpired:
        log(f"[WARN] timeout counting {marker} in {Path(file_path).name}")
        return 0

# ---------------------------------------------------------------------------
# State persistence (v3: remark-level flat dict)
# ---------------------------------------------------------------------------

def _make_key(rel_file: str, marker: str) -> str:
    return f"{rel_file}:{marker}"


class StateManager:
    def __init__(self, state_file: str):
        self._path = Path(state_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.state = self._load()

    def _load(self):
        if not self._path.exists():
            return {"version": 3, "results": {}}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if data.get("version") == 3:
                return data
            return {"version": 3, "results": {}}
        except (json.JSONDecodeError, OSError):
            return {"version": 3, "results": {}}

    def save(self):
        tmp = _state_tmp_path(self._path)
        with self._lock:
            tmp.write_text(json.dumps(self.state, indent=2,
                                      ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._path)

    def get(self, key: str):
        return self.state["results"].get(key)

    def upsert(self, key: str, entry: dict):
        with self._lock:
            self.state["results"][key] = entry

    def get_pending_keys(self):
        return [k for k, v in self.state["results"].items()
                if v.get("status") != "done"]

    def all_done_count(self):
        return sum(1 for v in self.state["results"].values()
                   if v.get("status") == "done")

    def total_count(self):
        return len(self.state["results"])

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


def item_status(entry: dict):
    if entry.get("count", 0) == 0:
        return "SKIP"
    if entry.get("failed", 0) > 0 or entry.get("error", 0) > 0:
        return "FAIL"
    return "PASS"


def format_done_line(entry: dict, done_count: int, total_items: int,
                     totals: dict, file_log_path: str = ""):
    status = item_status(entry)
    fname = Path(entry["file"]).name
    if entry.get("count", 0) == 0:
        result = "no tests collected"
    else:
        result = f"{entry.get('count', 0)} tests -> {format_result_line(entry)}"
    line = (
        f"{status:<5} [{entry['marker']}] {fname}  {result}  "
        f"{entry.get('duration', 0.0):.1f}s  |  "
        f"total items={done_count}/{total_items}  "
        f"passed={totals['passed']}  failed={totals['failed']}  "
        f"skipped={totals['skipped']}  error={totals['error']}  "
        f"time={totals['duration']:.1f}s"
    )
    if file_log_path:
        line += f"  log: {file_log_path}"
    elif status == "FAIL":
        line += "  log: pending file completion"
    return line


def run_work_item(abs_file: str, rel_file: str, marker: str, count: int,
                  xdist: int):
    fname = Path(abs_file).name

    if count == 0:
        entry = {"file": rel_file, "marker": marker, "count": 0,
                 "passed": 0, "failed": 0, "skipped": 0, "error": 0,
                 "duration": 0.0, "log": "", "exit_code": 0,
                 "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "status": "done"}
        return entry, ""

    log(f"START [{marker}] {fname}  ({count} tests)")

    cmd = [
        "pytest", abs_file,
        "-m", marker,
        "--tb=short", "-q",
        "-p", "no:cacheprovider",
    ]
    if xdist > 0:
        cmd.extend(["-n", str(xdist)])

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    env["COVERAGE_PROCESS_START"] = ""

    r = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=TRITON_REPO_DIR
    )

    duration = time.time() - t0
    summary = parse_pytest_summary(r.stdout + r.stderr)
    if r.returncode != 0 and summary["failed"] == 0 and summary["error"] == 0:
        summary["error"] = 1
    summary["duration"] = duration
    summary["exit_code"] = r.returncode

    has_fail = summary["failed"] > 0 or summary["error"] > 0

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = {
        "file": rel_file,
        "marker": marker,
        "count": count,
        "passed": summary["passed"],
        "failed": summary["failed"],
        "skipped": summary["skipped"],
        "error": summary["error"],
        "duration": summary["duration"],
        "log": "",
        "exit_code": summary.get("exit_code", 0),
        "timestamp": ts,
        "status": "done",
    }

    output = r.stdout + r.stderr if has_fail else ""
    return entry, output

# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.clear_cache:
        cleared = clear_script_cache(args.state_file)
        if cleared:
            print("Cleared script cache:")
            for path in cleared:
                print(f"  {path}")
        else:
            print(f"No script cache found for {args.state_file}")
        return

    if args.force:
        cleared = clear_script_cache(args.state_file)
        if cleared:
            print("Cleared script cache before forced run:")
            for path in cleared:
                print(f"  {path}")
        else:
            print(f"No script cache found for {args.state_file}; forced run will create it")
        if args.resume:
            print("  --force ignores --resume and runs all discovered items")
            args.resume = False

    state_mgr = StateManager(args.state_file)

    xdist = args.pytest_xdist

    # ── Phase 1: Discovery ────────────────────────────────────────
    print("── Discovery ──────────────────────────────────────────")

    files, global_excluded = discover_test_files(args.tests_dir, args.include, args.exclude)
    print(f"  Files:        {len(files)}")
    if global_excluded:
        print(f"  Global excl:  {len(global_excluded)} files")
    if not files:
        print("No test files found. Aborting.")
        sys.exit(1)

    file_markers = {}
    all_markers = set()
    for f in files:
        m = extract_markers(f)
        file_markers[f] = m
        all_markers.update(m)
    print(f"  Markers:      {len(all_markers)} unique")

    if args.markers:
        selected = {s.strip() for s in args.markers.split(",")}
        all_markers = selected & all_markers
        print(f"  (filtered to {len(all_markers)} user-specified)")

    pairs = []
    for f in files:
        for m in sorted(file_markers[f]):
            if m in all_markers:
                pairs.append((f, m))
    print(f"  Work items:   {len(pairs)}")

    # Count tests for pairs not yet cached in state
    need_count = []
    cached_count = 0
    for f, m in pairs:
        rf = rel_path(f, args.tests_dir)
        key = _make_key(rf, m)
        existing = state_mgr.get(key)
        if existing and "count" in existing:
            cached_count += 1
        else:
            need_count.append((f, m, rf))

    if need_count:
        print(f"  Counting {len(need_count)} items ({cached_count} cached)...  ",
              end="", flush=True)
        t0 = time.time()
        count_lock = threading.Lock()

        def _count(p):
            f, m, rf = p
            c = count_tests_for_item(f, m)
            key = _make_key(rf, m)
            entry = {"file": rf, "marker": m, "count": c,
                     "passed": 0, "failed": 0, "skipped": 0, "error": 0,
                     "duration": 0.0, "log": "", "exit_code": 0,
                     "timestamp": "", "status": "pending"}
            with count_lock:
                state_mgr.upsert(key, entry)

        n_cpu = min(8, len(need_count) or 1)
        with ThreadPoolExecutor(max_workers=n_cpu) as pool:
            pool.map(_count, need_count)

        state_mgr.save()
        print(f" done in {time.time()-t0:.1f}s")
    else:
        print(f"  All {len(pairs)} items cached, no counting needed")

    # ── Determine execution set ───────────────────────────────────
    if args.resume:
        exec_pairs = []
        for f, m in pairs:
            rf = rel_path(f, args.tests_dir)
            key = _make_key(rf, m)
            existing = state_mgr.get(key)
            if not existing or existing.get("status") != "done":
                exec_pairs.append((f, m, rf))
        if not exec_pairs:
            print("All items already completed. Nothing to do.")
            return
        print(f"\n── Resuming: {len(exec_pairs)} items to run "
              f"({len(pairs) - len(exec_pairs)} skipped) ──\n")
    else:
        exec_pairs = [(f, m, rel_path(f, args.tests_dir)) for f, m in pairs]
        print(f"\n── Execution ({len(exec_pairs)} items, "
              f"{args.workers} workers) ──\n")

    # ── Phase 2: Execution ────────────────────────────────────────

    done_count = 0
    dc_lock = threading.Lock()
    total_items = len(exec_pairs)
    file_total_items = {}
    for _, _, rf in exec_pairs:
        file_total_items[rf] = file_total_items.get(rf, 0) + 1
    file_done_items = {}
    file_failed_outputs = {}

    def flush_file_log_if_complete(rel_file: str):
        if file_done_items.get(rel_file, 0) != file_total_items.get(rel_file, 0):
            return ""

        failed_outputs = file_failed_outputs.pop(rel_file, [])
        if not failed_outputs:
            return ""

        logpath = save_file_log(rel_file, failed_outputs)
        for item in failed_outputs:
            entry = item["entry"]
            entry["log"] = logpath
            state_mgr.upsert(_make_key(entry["file"], entry["marker"]), entry)
        return logpath

    def on_item_done(future):
        nonlocal done_count
        try:
            entry, output = future.result()
        except Exception as e:
            log(f"ERROR item: {e}")
            with dc_lock:
                done_count += 1
            return

        rel_file = entry["file"]
        state_mgr.upsert(_make_key(rel_file, entry["marker"]), entry)

        if output:
            file_failed_outputs.setdefault(rel_file, []).append({
                "entry": entry,
                "output": output,
            })

        file_done_items[rel_file] = file_done_items.get(rel_file, 0) + 1
        file_log_path = flush_file_log_if_complete(rel_file)
        state_mgr.save()

        with dc_lock:
            done_count += 1
            cur = done_count
        s = state_mgr.summary_totals()
        log(format_done_line(entry, cur, total_items, s, file_log_path))

    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = []
    for f, m, rf in exec_pairs:
        key = _make_key(rf, m)
        entry = state_mgr.get(key)
        count = entry.get("count", 0) if entry else 0
        fut = pool.submit(run_work_item, f, rf, m, count, xdist)
        futures.append(fut)

    for fut in as_completed(futures):
        on_item_done(fut)

    pool.shutdown(wait=False)

    # ── Final summary ─────────────────────────────────────────────
    s = state_mgr.summary_totals()

    print()
    print("=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total items:  {state_mgr.total_count()}")
    print(f"  Done items:   {state_mgr.all_done_count()}")
    print(f"  Tests passed: {s['passed']}")
    print(f"  Tests failed: {s['failed']}")
    print(f"  Tests skipped:{s['skipped']}")
    print(f"  Tests error:  {s['error']}")
    print(f"  Total time:   {s['duration']:.1f}s")
    print("=" * 60)

    failed_entries = [v for v in state_mgr.state["results"].values()
                     if v.get("failed", 0) > 0 or v.get("error", 0) > 0]
    if failed_entries:
        print("\n  Failed items:")
        for v in failed_entries:
            fname = Path(v["file"]).name
            print(f"    {fname} - {v['marker']}  "
                  f"(passed={v['passed']} failed={v['failed']} "
                  f"skipped={v.get('skipped',0)})  "
                  f"duration={v.get('duration',0.0):.1f}s")
            if v.get("log"):
                print(f"      log: {v['log']}")

    if s["failed"] > 0 or s["error"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
