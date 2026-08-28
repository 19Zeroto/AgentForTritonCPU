#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


OVERHEAD_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)%\s+(.*)$")
COLUMN_SPLIT_RE = re.compile(r"\s{2,}")


@dataclass(frozen=True)
class PerfRow:
    overhead: float
    comm: str
    dso: str
    symbol: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a self-contained hotspot heatmap from a perf report, "
            "perf.data file, or FlagGems profile op/case directory."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a profile directory, perf_report.txt, or perf.data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML output path. Defaults to perf_heatmap.html next to the input.",
    )
    parser.add_argument(
        "--svg-output",
        type=Path,
        default=None,
        help="SVG output path. Defaults to perf_heatmap.svg next to the HTML.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Number of hottest rows to include. Defaults to 50.",
    )
    parser.add_argument(
        "--refresh-report",
        action="store_true",
        help="Regenerate perf_report.txt from perf.data even if it already exists.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title shown in the generated HTML/SVG.",
    )
    args = parser.parse_args()
    if args.top <= 0:
        parser.error("--top must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        report_path, output_dir = resolve_report(args.input, args.refresh_report)
        rows = parse_perf_report(report_path)
    except Exception as exc:
        print(f"perf_heatmap: {exc}", file=sys.stderr)
        return 2

    if not rows:
        print(f"perf_heatmap: no perf rows parsed from {report_path}", file=sys.stderr)
        return 1

    rows = sorted(rows, key=lambda row: row.overhead, reverse=True)[: args.top]
    html_path = args.output or (output_dir / "perf_heatmap.html")
    svg_path = args.svg_output or html_path.with_suffix(".svg")
    title = args.title or f"Perf Hotspot Heatmap: {output_dir.name}"

    svg = render_svg(rows, title=title, source=report_path)
    svg_path.write_text(svg, encoding="utf-8")
    html_path.write_text(
        render_html(rows, title=title, source=report_path, svg_path=svg_path),
        encoding="utf-8",
    )
    print(f"wrote {html_path}")
    print(f"wrote {svg_path}")
    return 0


def resolve_report(input_path: Path, refresh_report: bool) -> tuple[Path, Path]:
    path = input_path.expanduser().resolve()
    if path.is_dir():
        output_dir = path
        report_path = path / "perf_report.txt"
        data_path = path / "perf.data"
    elif path.name == "perf.data":
        output_dir = path.parent
        report_path = output_dir / "perf_report.txt"
        data_path = path
    else:
        output_dir = path.parent
        report_path = path
        data_path = output_dir / "perf.data"

    if report_path.exists() and not refresh_report:
        return report_path, output_dir
    if not data_path.exists():
        raise FileNotFoundError(
            f"no perf_report.txt found and perf.data is missing: {data_path}"
        )
    run_perf_report(data_path, report_path)
    return report_path, output_dir


def run_perf_report(data_path: Path, report_path: Path) -> None:
    if shutil.which("perf") is None:
        raise RuntimeError("perf is not available in PATH")
    command = [
        "perf",
        "report",
        "--stdio",
        "--input",
        str(data_path),
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
    report_path.write_text(completed.stdout or "", encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            f"perf report exited with code {completed.returncode}; see {report_path}"
        )


def parse_perf_report(report_path: Path) -> list[PerfRow]:
    rows: list[PerfRow] = []
    for line in report_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        match = OVERHEAD_RE.match(line)
        if not match:
            continue
        overhead = float(match.group(1))
        parts = COLUMN_SPLIT_RE.split(match.group(2).strip(), maxsplit=3)
        if len(parts) < 3:
            continue
        comm = parts[0].strip()
        dso = parts[1].strip()
        symbol = clean_symbol(parts[2].strip())
        if not symbol:
            continue
        rows.append(PerfRow(overhead=overhead, comm=comm, dso=dso, symbol=symbol))
    return rows


def clean_symbol(symbol: str) -> str:
    symbol = re.sub(r"\s+-\s+-\s*$", "", symbol)
    symbol = re.sub(r"\s+", " ", symbol)
    return symbol.strip()


def group_by_dso(rows: list[PerfRow]) -> list[tuple[str, float, list[PerfRow]]]:
    grouped: dict[str, list[PerfRow]] = {}
    for row in rows:
        grouped.setdefault(row.dso, []).append(row)
    result = []
    for dso, items in grouped.items():
        total = sum(item.overhead for item in items)
        result.append((dso, total, sorted(items, key=lambda row: row.overhead, reverse=True)))
    return sorted(result, key=lambda item: item[1], reverse=True)


def render_html(
    rows: list[PerfRow], *, title: str, source: Path, svg_path: Path
) -> str:
    generated = datetime.now().isoformat(timespec="seconds")
    max_overhead = max(row.overhead for row in rows)
    dso_groups = group_by_dso(rows)
    svg_rel = html.escape(svg_path.name)

    dso_cards = []
    for dso, total, items in dso_groups:
        cells = []
        for row in items:
            color = color_for(row.overhead, max_overhead)
            label = f"{row.overhead:.2f}% {row.symbol}"
            cells.append(
                "<div class=\"cell\" style=\"background:{color}\" title=\"{title}\">"
                "<span class=\"pct\">{pct:.2f}%</span>"
                "<span class=\"sym\">{sym}</span>"
                "</div>".format(
                    color=color,
                    title=html.escape(f"{row.comm} | {row.dso} | {row.symbol}"),
                    pct=row.overhead,
                    sym=html.escape(shorten(label, 84)),
                )
            )
        dso_cards.append(
            "<section class=\"group\">"
            "<h2>{dso} <span>{total:.2f}%</span></h2>"
            "<div class=\"cells\">{cells}</div>"
            "</section>".format(
                dso=html.escape(dso),
                total=total,
                cells="\n".join(cells),
            )
        )

    table_rows = []
    for index, row in enumerate(rows, start=1):
        table_rows.append(
            "<tr><td>{idx}</td><td>{pct:.2f}%</td><td>{comm}</td>"
            "<td>{dso}</td><td>{sym}</td></tr>".format(
                idx=index,
                pct=row.overhead,
                comm=html.escape(row.comm),
                dso=html.escape(row.dso),
                sym=html.escape(row.symbol),
            )
        )

    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
:root {{ color-scheme: light; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }}
body {{ margin: 0; background: #f5f6f8; color: #1f2933; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
.meta {{ color: #5b6472; margin-bottom: 24px; }}
.summary {{ display: grid; gap: 14px; }}
.group {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; }}
.group h2 {{ margin: 0 0 10px; font-size: 16px; display: flex; justify-content: space-between; gap: 16px; }}
.cells {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 8px; }}
.cell {{ min-height: 58px; border-radius: 6px; padding: 8px; color: #fff; box-shadow: inset 0 0 0 1px rgba(0,0,0,.12); overflow: hidden; }}
.pct {{ display: block; font-weight: 700; font-size: 15px; }}
.sym {{ display: block; margin-top: 4px; font-size: 12px; line-height: 1.25; overflow-wrap: anywhere; }}
.svg-link {{ margin: 18px 0; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d9dee7; }}
th, td {{ border-bottom: 1px solid #edf0f4; padding: 8px 10px; text-align: left; vertical-align: top; }}
th {{ background: #eef2f7; }}
td:nth-child(1), td:nth-child(2) {{ white-space: nowrap; text-align: right; }}
</style>
</head>
<body>
<main>
<h1>{title}</h1>
<div class="meta">Generated {generated}<br>Source: <code>{source}</code></div>
<div class="svg-link">SVG output: <a href="{svg_rel}">{svg_rel}</a></div>
<div class="summary">{groups}</div>
<h2>Top Rows</h2>
<table>
<thead><tr><th>#</th><th>Overhead</th><th>Command</th><th>DSO</th><th>Symbol</th></tr></thead>
<tbody>{table_rows}</tbody>
</table>
</main>
</body>
</html>
""".format(
        title=html.escape(title),
        generated=html.escape(generated),
        source=html.escape(str(source)),
        svg_rel=svg_rel,
        groups="\n".join(dso_cards),
        table_rows="\n".join(table_rows),
    )


def render_svg(rows: list[PerfRow], *, title: str, source: Path) -> str:
    width = 1200
    label_width = 270
    row_height = 38
    header_height = 74
    gap = 6
    dso_groups = group_by_dso(rows)
    height = header_height + len(dso_groups) * (row_height + gap) + 24
    max_overhead = max(row.overhead for row in rows)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f5f6f8"/>',
        f'<text x="24" y="30" font-family="sans-serif" font-size="22" font-weight="700" fill="#1f2933">{xml_escape(title)}</text>',
        f'<text x="24" y="55" font-family="sans-serif" font-size="12" fill="#5b6472">Source: {xml_escape(str(source))}</text>',
    ]

    y = header_height
    heat_width = width - label_width - 48
    for dso, total, items in dso_groups:
        parts.append(
            f'<text x="24" y="{y + 24}" font-family="sans-serif" font-size="12" fill="#1f2933">'
            f"{xml_escape(shorten(dso, 34))} ({total:.2f}%)</text>"
        )
        x = label_width
        remaining = heat_width
        group_total = sum(item.overhead for item in items) or 1.0
        for index, row in enumerate(items):
            if index == len(items) - 1:
                cell_width = remaining
            else:
                cell_width = max(3, heat_width * row.overhead / group_total)
                remaining -= cell_width
            color = color_for(row.overhead, max_overhead)
            title_text = (
                f"{row.overhead:.2f}% | {row.comm} | {row.dso} | {row.symbol}"
            )
            parts.append(
                f'<rect x="{x:.2f}" y="{y}" width="{cell_width:.2f}" height="{row_height}" '
                f'rx="4" fill="{color}"><title>{xml_escape(title_text)}</title></rect>'
            )
            if cell_width > 82:
                parts.append(
                    f'<text x="{x + 6:.2f}" y="{y + 23}" font-family="sans-serif" '
                    f'font-size="11" fill="#fff">{xml_escape(shorten(row.symbol, int(cell_width / 7)))}</text>'
                )
            x += cell_width
        y += row_height + gap

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def color_for(value: float, max_value: float) -> str:
    t = 0.0 if max_value <= 0 else max(0.0, min(1.0, value / max_value))
    if t < 0.5:
        local = t / 0.5
        start = (43, 108, 176)
        end = (253, 174, 97)
    else:
        local = (t - 0.5) / 0.5
        start = (253, 174, 97)
        end = (215, 48, 39)
    r = round(start[0] + (end[0] - start[0]) * local)
    g = round(start[1] + (end[1] - start[1]) * local)
    b = round(start[2] + (end[2] - start[2]) * local)
    return f"rgb({r},{g},{b})"


def shorten(value: str, limit: int) -> str:
    if limit <= 3 or len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
