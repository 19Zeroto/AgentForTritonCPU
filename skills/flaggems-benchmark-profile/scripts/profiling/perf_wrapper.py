from __future__ import annotations

import csv
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PerfResult:
    events: dict[str, float] = field(default_factory=dict)
    raw_rows: list[list[str]] = field(default_factory=list)


@dataclass
class PerfRecordResult:
    report_returncode: int | None = None
    report_command: list[str] = field(default_factory=list)
    top10_rows: list[str] = field(default_factory=list)
    note: str | None = None


class PerfStatRunner:
    def __init__(self, events: list[str], output_csv: str | Path):
        self.events = events
        self.output_csv = str(output_csv)

    @staticmethod
    def is_available() -> bool:
        return shutil.which("perf") is not None

    def build_command(self, target_cmd: list[str]) -> list[str]:
        return [
            "perf",
            "stat",
            "-e",
            ",".join(self.events),
            "-o",
            self.output_csv,
            "-x",
            ",",
            "--",
            *target_cmd,
        ]

    @staticmethod
    def parse_output(csv_path: str | Path) -> PerfResult:
        path = Path(csv_path)
        if not path.exists():
            return PerfResult()

        result = PerfResult()
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or row[0].lstrip().startswith("#"):
                    continue
                result.raw_rows.append(row)
                parsed = _parse_perf_row(row)
                if parsed is None:
                    continue
                event, value = parsed
                result.events[event] = value
        return result


def normalize_event_name(event: str) -> str:
    return (
        event.strip()
        .replace("-", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(".", "_")
    )


def derive_cache_metrics(events: dict[str, float]) -> dict[str, float]:
    normalized = {normalize_event_name(key): value for key, value in events.items()}
    derived: dict[str, float] = {}

    l1_loads = normalized.get("L1_dcache_loads")
    l1_misses = normalized.get("L1_dcache_load_misses")
    if l1_loads:
        derived["L1_dcache_miss_rate"] = (l1_misses or 0.0) / l1_loads

    llc_loads = normalized.get("LLC_loads")
    llc_misses = normalized.get("LLC_load_misses")
    if llc_loads:
        derived["LLC_miss_rate"] = (llc_misses or 0.0) / llc_loads

    instructions = normalized.get("instructions")
    cycles = normalized.get("cycles")
    if cycles:
        derived["ipc"] = (instructions or 0.0) / cycles

    return {**normalized, **derived}


def _parse_perf_row(row: list[str]) -> tuple[str, float] | None:
    if len(row) < 3:
        return None
    value = _parse_number(row[0])
    if value is None:
        return None

    event = row[2].strip() if len(row) > 2 else ""
    if not event and len(row) > 1:
        event = row[1].strip()
    if not event:
        return None
    return event, value


def _parse_number(value: str) -> float | None:
    text = value.strip()
    if not text or text.startswith("<"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


class PerfRecordRunner:
    def __init__(
        self,
        output_data: str | Path,
        report_txt: str | Path,
        top10_txt: str | Path,
        *,
        frequency: int = 99,
        call_graph: str = "dwarf",
        top_limit: int = 10,
    ):
        self.output_data = Path(output_data)
        self.report_txt = Path(report_txt)
        self.top10_txt = Path(top10_txt)
        self.frequency = frequency
        self.call_graph = call_graph
        self.top_limit = top_limit

    @staticmethod
    def is_available() -> bool:
        return shutil.which("perf") is not None

    def build_command(self, target_cmd: list[str]) -> list[str]:
        command = [
            "perf",
            "record",
            "-F",
            str(self.frequency),
            "-o",
            str(self.output_data),
        ]
        if self.call_graph != "none":
            command.extend(["-g", "--call-graph", self.call_graph])
        command.extend(["--", *target_cmd])
        return command

    def write_reports(self) -> PerfRecordResult:
        if not self.output_data.exists():
            note = f"perf record data not found: {self.output_data}"
            self.report_txt.write_text(note + "\n", encoding="utf-8")
            self.top10_txt.write_text(note + "\n", encoding="utf-8")
            return PerfRecordResult(note=note)

        command = [
            "perf",
            "report",
            "--stdio",
            "--input",
            str(self.output_data),
            "--no-children",
            "--sort",
            "comm,dso,symbol",
        ]
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        report_text = completed.stdout or ""
        self.report_txt.write_text(report_text, encoding="utf-8", errors="replace")

        rows = extract_perf_top_rows(report_text, self.top_limit)
        if rows:
            top_text = "\n".join(rows) + "\n"
            note = None
        else:
            top_text = (
                "No perf report rows were parsed. See perf_report.txt for the raw "
                "report output.\n"
            )
            note = "perf report produced no parsable top rows"
        self.top10_txt.write_text(top_text, encoding="utf-8")

        if completed.returncode != 0:
            note = (
                f"perf report exited with code {completed.returncode}; "
                "see perf_report.txt"
            )
        return PerfRecordResult(
            report_returncode=completed.returncode,
            report_command=command,
            top10_rows=rows,
            note=note,
        )


def extract_perf_top_rows(report_text: str, limit: int = 10) -> list[str]:
    rows: list[str] = []
    overhead_row = re.compile(r"^\s*\d+(?:\.\d+)?%")
    for line in report_text.splitlines():
        if overhead_row.match(line):
            rows.append(line.rstrip())
            if len(rows) >= limit:
                break
    return rows
