#!/usr/bin/env python3
"""Analyze period-weighted kernel hotspots in one or more perf.data files.

Usage:
  python3 analyze_perf.py <case-dir-or-perf.data> \
    --kernel-symbol <symbol> --output-dir <external-dir>

Example:
  python3 analyze_perf.py "$CASE/perf.data" \
    --kernel-symbol dreglu_kernel --child-symbol memrefCopy \
    --output-dir "$AGENT_DIR/logs/kernel-perf/dreglu-run-a"

Design and interpretation rules:
  ../references/workflow.md and ../references/interpretation.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PERCENT_RE = re.compile(r"^(\d+(?:\.\d+)?)%$")
ANNOTATION_RE = re.compile(
    r"^\s*([0-9][0-9,]*)\s*:\s*([0-9a-fA-F]+):\s*(\S+)(?:\s+(.*?))?\s*$"
)
EVENT_RE = re.compile(r"Samples:\s+.*?event '([^']+)'", re.MULTILINE)
EVENT_COUNT_RE = re.compile(r"Event count \(approx\.\):\s*([0-9,]+)")
LOST_SAMPLES_RE = re.compile(r"Total Lost Samples:\s*([0-9,]+)")


@dataclass(frozen=True)
class PerfRow:
    symbol: str
    dso: str
    self_percent: float
    period: int | None = None
    children_percent: float | None = None


@dataclass(frozen=True)
class InstructionRow:
    period: int
    address: str
    mnemonic: str
    operands: str
    family: str

    @property
    def instruction(self) -> str:
        return f"{self.mnemonic} {self.operands}".strip()


@dataclass
class SymbolAnalysis:
    symbol: str
    annotation_path: str
    total_period: int
    process_self_percent: float | None
    process_self_period: int | None
    process_children_percent: float | None
    family_periods: dict[str, int]
    mnemonic_periods: dict[str, int]
    top_instructions: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run perf report/annotate on existing perf.data and generate a "
            "period-weighted kernel bottleneck report."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more case directories or perf.data files for repeat analysis.",
    )
    parser.add_argument(
        "--kernel-symbol",
        required=True,
        help="Exact generated kernel symbol from perf report.",
    )
    parser.add_argument(
        "--child-symbol",
        action="append",
        default=[],
        help=(
            "Child symbol to annotate, repeatable. Only pass symbols whose call "
            "chain has already been confirmed under the kernel."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "External result directory. Defaults to "
            "$AGENT_DIR/logs/AgentForTritonCPU/flaggems-kernel-perf/<timestamp>."
        ),
    )
    parser.add_argument(
        "--shape",
        default=None,
        help="Optional shape label included in the generated report.",
    )
    parser.add_argument(
        "--objdump",
        type=Path,
        default=None,
        help=(
            "Objdump passed to perf annotate. Defaults to "
            "$LLVM_INSTALL_DIR/bin/llvm-objdump when available."
        ),
    )
    parser.add_argument(
        "--perf",
        default="perf",
        help="perf executable name or path. Defaults to perf.",
    )
    parser.add_argument(
        "--top-instructions",
        type=int,
        default=12,
        help="Number of period-heavy instructions to report. Defaults to 12.",
    )
    parser.add_argument(
        "--top-process-symbols",
        type=int,
        default=12,
        help="Number of process-self symbols to report. Defaults to 12.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse generated raw perf text files in --output-dir when present.",
    )
    args = parser.parse_args()
    if args.top_instructions <= 0:
        parser.error("--top-instructions must be positive")
    if args.top_process_symbols <= 0:
        parser.error("--top-process-symbols must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        perf_executable = resolve_executable(args.perf)
        objdump = resolve_objdump(args.objdump)
        output_root = resolve_output_dir(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        perf_data_files = [resolve_perf_data(path) for path in args.inputs]
        analyses = []
        for index, perf_data in enumerate(perf_data_files, start=1):
            run_dir = output_root if len(perf_data_files) == 1 else output_root / f"run_{index:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            analyses.append(
                analyze_run(
                    perf_data=perf_data,
                    output_dir=run_dir,
                    perf_executable=perf_executable,
                    objdump=objdump,
                    kernel_symbol=args.kernel_symbol,
                    child_symbols=deduplicate(args.child_symbol),
                    shape=args.shape,
                    top_instructions=args.top_instructions,
                    top_process_symbols=args.top_process_symbols,
                    reuse=args.reuse,
                )
            )
        if len(analyses) > 1:
            write_repeat_comparison(output_root, analyses, args.kernel_symbol, args.shape)
    except Exception as exc:
        print(f"analyze_perf: {exc}", file=sys.stderr)
        return 2

    print(f"wrote analysis to {output_root}")
    return 0


def resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.parent != Path("."):
        if not candidate.is_file():
            raise FileNotFoundError(f"perf executable not found: {candidate}")
        return str(candidate.resolve())
    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(f"perf executable not found in PATH: {value}")
    return resolved


def resolve_objdump(explicit: Path | None) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"objdump not found: {path}")
        return path

    llvm_install = os.environ.get("LLVM_INSTALL_DIR")
    if llvm_install:
        path = Path(llvm_install).expanduser() / "bin" / "llvm-objdump"
        if path.is_file():
            return path.resolve()
    discovered = shutil.which("llvm-objdump")
    return Path(discovered).resolve() if discovered else None


def resolve_output_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        result = explicit.expanduser().resolve()
    else:
        agent_dir = os.environ.get("AGENT_DIR")
        if agent_dir:
            workspace = Path(agent_dir).expanduser().resolve()
        else:
            workspace = Path(__file__).resolve().parents[4]
        stamp = datetime.now().strftime("analysis_%Y%m%d_%H%M%S")
        result = (
            workspace
            / "logs"
            / "AgentForTritonCPU"
            / "flaggems-kernel-perf"
            / stamp
        )

    source_roots = [Path(__file__).resolve().parents[3]]
    triton_repo = os.environ.get("TRITON_REPO_DIR")
    if triton_repo:
        source_roots.append(Path(triton_repo).expanduser().resolve())
    for source_root in source_roots:
        if result == source_root or source_root in result.parents:
            raise ValueError(
                f"output directory must be outside source repositories: {result}"
            )
    return result


def resolve_perf_data(value: Path) -> Path:
    path = value.expanduser().resolve()
    if path.is_dir():
        path = path / "perf.data"
    if not path.is_file():
        raise FileNotFoundError(f"perf.data not found: {path}")
    return path


def analyze_run(
    *,
    perf_data: Path,
    output_dir: Path,
    perf_executable: str,
    objdump: Path | None,
    kernel_symbol: str,
    child_symbols: list[str],
    shape: str | None,
    top_instructions: int,
    top_process_symbols: int,
    reuse: bool,
) -> dict[str, Any]:
    commands: list[list[str]] = []
    metadata_path = output_dir / "perf_metadata.txt"
    flat_path = output_dir / "perf_report_flat.txt"
    callgraph_path = output_dir / "perf_report_callgraph.txt"

    metadata = run_or_reuse(
        [perf_executable, "evlist", "-v", "-i", str(perf_data)],
        metadata_path,
        commands,
        reuse,
    )
    flat_text = run_or_reuse(
        [
            perf_executable,
            "report",
            "--stdio",
            "--input",
            str(perf_data),
            "--no-children",
            "--show-total-period",
            "--sort",
            "dso,symbol",
        ],
        flat_path,
        commands,
        reuse,
    )
    callgraph_text = run_or_reuse(
        [
            perf_executable,
            "report",
            "--stdio",
            "--input",
            str(perf_data),
            "--sort",
            "dso,symbol",
        ],
        callgraph_path,
        commands,
        reuse,
    )

    flat_rows = parse_flat_report(flat_text)
    callgraph_rows = parse_callgraph_report(callgraph_text)
    has_callchain = "CALLCHAIN" in metadata
    kernel_flat = find_symbol(flat_rows, kernel_symbol)
    kernel_callgraph = find_symbol(callgraph_rows, kernel_symbol)
    if kernel_flat is None:
        raise RuntimeError(
            f"kernel symbol {kernel_symbol!r} was not found in {flat_path}; "
            "use the exact symbol from perf_report_flat.txt"
        )

    symbol_analyses: list[SymbolAnalysis] = []
    for symbol in [kernel_symbol, *child_symbols]:
        safe_symbol = safe_name(symbol)
        annotation_path = output_dir / f"perf_annotate_{safe_symbol}_period.txt"
        command = [
            perf_executable,
            "annotate",
            "--stdio",
            "--input",
            str(perf_data),
            "--symbol",
            symbol,
            "--percent-type",
            "local-period",
            "--show-total-period",
            "--full-paths",
        ]
        if objdump is not None:
            command[3:3] = ["--objdump", str(objdump)]
        annotation_text = run_or_reuse(
            command, annotation_path, commands, reuse
        )
        instructions = parse_annotation(annotation_text)
        if not instructions:
            raise RuntimeError(
                f"no period-weighted instructions parsed for {symbol!r}; "
                f"see {annotation_path}"
            )
        flat_row = find_symbol(flat_rows, symbol)
        callgraph_row = find_symbol(callgraph_rows, symbol)
        symbol_analysis = summarize_symbol(
            symbol=symbol,
            annotation_path=annotation_path,
            instructions=instructions,
            flat_row=flat_row,
            callgraph_row=callgraph_row if has_callchain else None,
            top_instructions=top_instructions,
        )
        symbol_analyses.append(symbol_analysis)
        write_instruction_csv(
            output_dir / f"instruction_hotspots_{safe_symbol}.csv",
            instructions,
        )
        write_mnemonic_csv(
            output_dir / f"mnemonic_summary_{safe_symbol}.csv",
            symbol_analysis,
        )

    event = first_match(EVENT_RE, flat_text)
    event_count = parse_optional_int(first_match(EVENT_COUNT_RE, flat_text))
    lost_samples = parse_optional_int(first_match(LOST_SAMPLES_RE, flat_text))
    command_lines = [shlex.join(command) for command in commands]
    (output_dir / "commands.txt").write_text(
        "\n".join(command_lines) + "\n", encoding="utf-8"
    )

    result = {
        "perf_data": str(perf_data),
        "shape": shape,
        "event": event,
        "event_count": event_count,
        "lost_samples": lost_samples,
        "callchain_recorded": has_callchain,
        "kernel_symbol": kernel_symbol,
        "kernel_process_self_percent": kernel_flat.self_percent,
        "kernel_process_self_period": kernel_flat.period,
        "kernel_process_children_percent": (
            kernel_callgraph.children_percent
            if has_callchain and kernel_callgraph is not None
            else None
        ),
        "outside_kernel_tree_percent": outside_percent(
            kernel_callgraph.children_percent
            if has_callchain and kernel_callgraph is not None
            else None
        ),
        "symbols": [asdict(item) for item in symbol_analyses],
        "top_process_self_symbols": [
            asdict(row)
            for row in sorted(
                flat_rows, key=lambda item: item.self_percent, reverse=True
            )[:top_process_symbols]
        ],
        "commands": command_lines,
        "raw_files": {
            "metadata": str(metadata_path),
            "flat_report": str(flat_path),
            "callgraph_report": str(callgraph_path),
        },
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        render_report(result), encoding="utf-8"
    )
    return result


def run_or_reuse(
    command: list[str],
    output_path: Path,
    commands: list[list[str]],
    reuse: bool,
) -> str:
    commands.append(command)
    if reuse and output_path.is_file():
        return output_path.read_text(encoding="utf-8", errors="replace")

    env = os.environ.copy()
    env["PERF_PAGER"] = "cat"
    env["PAGER"] = "cat"
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
    )
    output = completed.stdout or ""
    output_path.write_text(output, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command exited with code {completed.returncode}: "
            f"{shlex.join(command)}; see {output_path}"
        )
    return output


def parse_flat_report(text: str) -> list[PerfRow]:
    rows: list[PerfRow] = []
    for line in text.splitlines():
        parts = split_columns(line)
        if len(parts) < 4 or PERCENT_RE.fullmatch(parts[0]) is None:
            continue
        period_text = parts[1].replace(",", "")
        if not period_text.isdigit():
            continue
        rows.append(
            PerfRow(
                symbol=clean_symbol(parts[3]),
                dso=parts[2],
                self_percent=parse_percent(parts[0]),
                period=int(period_text),
            )
        )
    return rows


def parse_callgraph_report(text: str) -> list[PerfRow]:
    rows: list[PerfRow] = []
    for line in text.splitlines():
        parts = split_columns(line)
        if (
            len(parts) < 4
            or PERCENT_RE.fullmatch(parts[0]) is None
            or PERCENT_RE.fullmatch(parts[1]) is None
        ):
            continue
        rows.append(
            PerfRow(
                symbol=clean_symbol(parts[3]),
                dso=parts[2],
                children_percent=parse_percent(parts[0]),
                self_percent=parse_percent(parts[1]),
            )
        )
    return rows


def split_columns(line: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s{2,}", line.strip())]


def clean_symbol(value: str) -> str:
    value = re.sub(r"^\[[^]]+\]\s*", "", value)
    return value.strip()


def find_symbol(rows: list[PerfRow], symbol: str) -> PerfRow | None:
    exact = [row for row in rows if row.symbol == symbol]
    if not exact:
        return None
    return max(exact, key=lambda row: row.self_percent)


def parse_annotation(text: str) -> list[InstructionRow]:
    rows: list[InstructionRow] = []
    for line in text.splitlines():
        match = ANNOTATION_RE.match(line)
        if match is None:
            continue
        period = int(match.group(1).replace(",", ""))
        mnemonic = match.group(3).strip()
        operands = (match.group(4) or "").strip()
        rows.append(
            InstructionRow(
                period=period,
                address=match.group(2),
                mnemonic=mnemonic,
                operands=operands,
                family=instruction_family(mnemonic),
            )
        )
    return rows


def instruction_family(mnemonic: str) -> str:
    value = mnemonic.lower()
    if value.startswith("ld"):
        return "load"
    if value.startswith("st"):
        return "store"
    if value in {"bl", "blr"}:
        return "call-site"
    if value in {"b", "br", "ret", "cbz", "cbnz", "tbz", "tbnz"} or value.startswith("b."):
        return "branch"
    if value in {
        "add",
        "adds",
        "addvl",
        "sub",
        "subs",
        "mul",
        "madd",
        "msub",
        "lsl",
        "lsr",
        "asr",
        "adr",
        "adrp",
        "mov",
        "movk",
        "movn",
        "movz",
    }:
        return "address/index"
    return "compute/other"


def summarize_symbol(
    *,
    symbol: str,
    annotation_path: Path,
    instructions: list[InstructionRow],
    flat_row: PerfRow | None,
    callgraph_row: PerfRow | None,
    top_instructions: int,
) -> SymbolAnalysis:
    total_period = sum(item.period for item in instructions)
    if total_period <= 0:
        raise RuntimeError(f"annotation period total is zero for {symbol!r}")
    family_periods: dict[str, int] = {}
    mnemonic_periods: dict[str, int] = {}
    for item in instructions:
        family_periods[item.family] = family_periods.get(item.family, 0) + item.period
        mnemonic_periods[item.mnemonic] = mnemonic_periods.get(item.mnemonic, 0) + item.period

    top_rows = sorted(instructions, key=lambda item: item.period, reverse=True)[
        :top_instructions
    ]
    top_payload = []
    for item in top_rows:
        payload = asdict(item)
        payload["instruction"] = item.instruction
        payload["local_percent"] = percent(item.period, total_period)
        top_payload.append(payload)

    return SymbolAnalysis(
        symbol=symbol,
        annotation_path=str(annotation_path),
        total_period=total_period,
        process_self_percent=flat_row.self_percent if flat_row else None,
        process_self_period=flat_row.period if flat_row else None,
        process_children_percent=(
            callgraph_row.children_percent if callgraph_row else None
        ),
        family_periods=dict(
            sorted(family_periods.items(), key=lambda item: item[1], reverse=True)
        ),
        mnemonic_periods=dict(
            sorted(mnemonic_periods.items(), key=lambda item: item[1], reverse=True)
        ),
        top_instructions=top_payload,
    )


def write_instruction_csv(path: Path, rows: list[InstructionRow]) -> None:
    total = sum(item.period for item in rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["address", "instruction", "family", "period", "local_percent"]
        )
        for item in sorted(rows, key=lambda row: row.period, reverse=True):
            writer.writerow(
                [
                    item.address,
                    item.instruction,
                    item.family,
                    item.period,
                    f"{percent(item.period, total):.6f}",
                ]
            )


def write_mnemonic_csv(path: Path, analysis: SymbolAnalysis) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mnemonic", "period", "local_percent"])
        for mnemonic, period_value in analysis.mnemonic_periods.items():
            writer.writerow(
                [
                    mnemonic,
                    period_value,
                    f"{percent(period_value, analysis.total_period):.6f}",
                ]
            )


def render_report(result: dict[str, Any]) -> str:
    kernel = next(
        item for item in result["symbols"] if item["symbol"] == result["kernel_symbol"]
    )
    lines = [
        "# FlagGems kernel perf bottleneck report",
        "",
        f"- perf.data: `{result['perf_data']}`",
        f"- Shape: `{result['shape'] or 'N/A'}`",
        f"- Event: `{result['event'] or 'N/A'}`",
        f"- Approximate event count: `{format_optional_int(result['event_count'])}`",
        f"- Lost samples: `{format_optional_int(result['lost_samples'])}`",
        f"- Call chain recorded: `{'yes' if result['callchain_recorded'] else 'no'}`",
        "",
        "## Attribution",
        "",
        "| Scope | Symbol | Period/share | Denominator | Meaning |",
        "|---|---|---:|---|---|",
        (
            f"| Process self | `{result['kernel_symbol']}` | "
            f"{format_percent(result['kernel_process_self_percent'])} | All recorded process periods | "
            "Instructions executed directly in the kernel symbol |"
        ),
        (
            f"| Process inclusive | `{result['kernel_symbol']}` | "
            f"{format_percent(result['kernel_process_children_percent'])} | All recorded process periods | "
            "Kernel self plus descendants; N/A without CALLCHAIN |"
        ),
        (
            "| Outside kernel tree | — | "
            f"{format_percent(result['outside_kernel_tree_percent'])} | All recorded process periods | "
            "Cannot be decomposed from flat rows alone |"
        ),
        (
            f"| Symbol local | `{result['kernel_symbol']}` | "
            f"{kernel['total_period']:,} period | Kernel self periods only | "
            "Denominator for the instruction tables below |"
        ),
        "",
        "## Kernel instruction families",
        "",
        "| Family | Period | Kernel-local share |",
        "|---|---:|---:|",
    ]
    for family, period_value in kernel["family_periods"].items():
        lines.append(
            f"| {family} | {period_value:,} | "
            f"{percent(period_value, kernel['total_period']):.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Hottest kernel instructions",
            "",
            "| Address | Instruction | Type | Period | Kernel-local share |",
            "|---|---|---|---:|---:|",
        ]
    )
    for item in kernel["top_instructions"]:
        lines.append(
            f"| `{item['address']}` | `{escape_pipe(item['instruction'])}` | "
            f"{item['family']} | {item['period']:,} | {item['local_percent']:.2f}% |"
        )

    child_items = [
        item for item in result["symbols"] if item["symbol"] != result["kernel_symbol"]
    ]
    if child_items:
        lines.extend(
            [
                "",
                "## Requested child-symbol annotations",
                "",
                "These symbols were explicitly requested for annotation. Confirm their "
                "parentage in `perf_report_callgraph.txt`; their local instruction shares "
                "use each child symbol's own period denominator.",
            ]
        )
        for child in child_items:
            lines.extend(
                [
                    "",
                    f"### `{child['symbol']}`",
                    "",
                    "| Family | Period | Child-local share |",
                    "|---|---:|---:|",
                ]
            )
            for family, period_value in child["family_periods"].items():
                lines.append(
                    f"| {family} | {period_value:,} | "
                    f"{percent(period_value, child['total_period']):.2f}% |"
                )

    lines.extend(
        [
            "",
            "## Process-self hotspots",
            "",
            "These rows are disjoint self-attribution, but a row may be inside or outside "
            "the kernel call tree. Use the call graph before assigning it to the remainder.",
            "",
            "| Symbol | DSO | Period | Process-self share |",
            "|---|---|---:|---:|",
        ]
    )
    for item in result["top_process_self_symbols"]:
        lines.append(
            f"| `{escape_pipe(item['symbol'])}` | `{escape_pipe(item['dso'])}` | "
            f"{format_optional_int(item['period'])} | {item['self_percent']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Percentages from annotation are period-weighted local shares, not static "
            "instruction counts and not raw sample-count shares.",
            "- With a cycles event, period is a sampled CPU-cycle weight and therefore a "
            "CPU-time proxy, not an exact wall-clock duration.",
            "- A callee's periods belong to the callee's Self and to the caller's inclusive "
            "Children. The call-site instruction itself does not absorb the callee cost.",
            "- Do not add Children and Self; Children already includes Self.",
            "- Do not call a kernel memory-bound from load/store share alone; corroborate "
            "with cache, bandwidth, IPC, or controlled variants.",
            "",
            "## Commands",
            "",
            "```sh",
            *result["commands"],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_repeat_comparison(
    output_root: Path,
    analyses: list[dict[str, Any]],
    kernel_symbol: str,
    shape: str | None,
) -> None:
    lines = [
        "# Repeat comparison",
        "",
        f"- Kernel: `{kernel_symbol}`",
        f"- Shape: `{shape or 'N/A'}`",
        "- All runs must use the same event, frequency, call-graph mode, benchmark "
        "arguments, affinity, and cache policy before the numbers are treated as repeats.",
        "",
        "| Run | Event | CALLCHAIN | Kernel self/process | Kernel inclusive/process | "
        "Outside tree | Load/local | Store/local |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    events = set()
    callchain_values = set()
    for index, result in enumerate(analyses, start=1):
        events.add(result.get("event"))
        callchain_values.add(result.get("callchain_recorded"))
        kernel = next(
            item for item in result["symbols"] if item["symbol"] == kernel_symbol
        )
        load_share = percent(
            kernel["family_periods"].get("load", 0), kernel["total_period"]
        )
        store_share = percent(
            kernel["family_periods"].get("store", 0), kernel["total_period"]
        )
        lines.append(
            f"| run_{index:02d} | `{result.get('event') or 'N/A'}` | "
            f"{'yes' if result.get('callchain_recorded') else 'no'} | "
            f"{format_percent(result.get('kernel_process_self_percent'))} | "
            f"{format_percent(result.get('kernel_process_children_percent'))} | "
            f"{format_percent(result.get('outside_kernel_tree_percent'))} | "
            f"{load_share:.2f}% | {store_share:.2f}% |"
        )

    lines.extend(["", "## Comparability", ""])
    if len(events) == 1 and len(callchain_values) == 1:
        lines.append(
            "Event and CALLCHAIN presence match. Verify the commands in each run before "
            "accepting repeat consistency."
        )
    else:
        lines.append(
            "**Not directly comparable:** event or CALLCHAIN presence differs across runs."
        )
    lines.append("")
    (output_root / "repeat_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def parse_optional_int(value: str | None) -> int | None:
    return int(value.replace(",", "")) if value is not None else None


def parse_percent(value: str) -> float:
    return float(value.rstrip("%"))


def percent(value: int | float, total: int | float) -> float:
    return 100.0 * value / total if total else 0.0


def outside_percent(children_percent: float | None) -> float | None:
    if children_percent is None:
        return None
    return max(0.0, 100.0 - children_percent)


def format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def format_optional_int(value: int | None) -> str:
    return "N/A" if value is None else f"{value:,}"


def escape_pipe(value: str) -> str:
    return value.replace("|", "\\|")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "symbol"


def deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


if __name__ == "__main__":
    raise SystemExit(main())
