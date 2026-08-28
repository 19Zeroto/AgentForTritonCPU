#!/usr/bin/env python3
"""
Summarize results from the v3 state format produced by
scripts/run_triton_tests.py.

Outputs a table grouped by file, then lists failed items with details.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AGENTFORTRITONCPU_DIR = SCRIPT_DIR.parent.parent.parent
AGENT_DIR = Path(
    os.environ.get("AGENT_DIR", AGENTFORTRITONCPU_DIR.parent)
).expanduser().resolve()
DEFAULT_STATE_FILE = str(
    AGENT_DIR
    / "logs"
    / "AgentForTritonCPU"
    / "test-suite"
    / "triton-test-state.json"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize triton test results")
    parser.add_argument("-s", "--state-file", default=DEFAULT_STATE_FILE,
                        help=f"State file to read (default: {DEFAULT_STATE_FILE})")
    return parser.parse_args(argv)


def fmt_dur(secs: float) -> str:
    return f"{secs:.1f}s"


def main():
    args = parse_args()
    path = Path(args.state_file)
    if not path.exists():
        print(f"State file not found: {path}")
        return

    data = json.loads(path.read_text())
    results = data.get("results", {})
    if not results:
        print("No results in state file.")
        return

    # Group entries by file
    by_file = defaultdict(list)
    for key, entry in results.items():
        by_file[entry["file"]].append(entry)

    all_passed = True
    rows = []

    for file in sorted(by_file):
        entries = by_file[file]
        total = sum(e.get("count", 0) for e in entries)
        passed = sum(e.get("passed", 0) for e in entries)
        failed = sum(e.get("failed", 0) for e in entries)
        skipped = sum(e.get("skipped", 0) for e in entries)
        duration = sum(e.get("duration", 0.0) for e in entries)
        fname = Path(file).name
        rows.append((fname, duration, total, passed, failed, skipped))
        if failed > 0:
            all_passed = False

    # Column widths
    col_w = [max(len(r[0]) for r in rows), 10, 7, 7, 7, 9]
    hdr_fmt = "  ".join(
        ["{{:<{}}}".format(col_w[0])] +
        ["{{:>{}}}".format(w) for w in col_w[1:]]
    )
    sep = "  ".join(
        ["\u2500" * col_w[0]] +
        ["\u2500" * w for w in col_w[1:]]
    )

    header = hdr_fmt.format("File", "Duration", "Tests", "Passed", "Failed", "Skipped")
    print()
    print(header)
    print(sep)

    for fname, duration, total, passed, failed, skipped in rows:
        print(hdr_fmt.format(
            fname, fmt_dur(duration), str(total),
            str(passed), str(failed), str(skipped),
        ))

    # Total row
    tot_dur = sum(r[1] for r in rows)
    tot_total = sum(r[2] for r in rows)
    tot_passed = sum(r[3] for r in rows)
    tot_failed = sum(r[4] for r in rows)
    tot_skipped = sum(r[5] for r in rows)

    print(sep)
    print(hdr_fmt.format(
        "Total", fmt_dur(tot_dur), str(tot_total),
        str(tot_passed), str(tot_failed), str(tot_skipped),
    ))
    print()

    if all_passed:
        print("All tests passed!")
        return

    # Failed items detail
    print("Failed items:")
    for key, entry in results.items():
        failed = entry.get("failed", 0)
        if failed == 0:
            continue
        fname = Path(entry["file"]).name
        marker = entry["marker"]
        total = entry.get("count", 0)
        dur = entry.get("duration", 0.0)
        print(f"  {fname} - {marker}    failed={failed}/{total}  duration={fmt_dur(dur)}")
        if entry.get("log"):
            print(f"    log: {entry['log']}")


if __name__ == "__main__":
    main()
