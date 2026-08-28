from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


SYSTEM_COMMANDS = [
    ("uname", ["uname", "-a"]),
    ("lscpu", ["lscpu"]),
    ("numactl", ["numactl", "-H"]),
    ("free", ["free", "-h"]),
]


def collect_system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "summary": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "commands": {},
        "files": {},
    }
    for name, command in SYSTEM_COMMANDS:
        info["commands"][name] = _run_command(command)

    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        try:
            info["files"]["/proc/cpuinfo"] = cpuinfo_path.read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError as exc:
            info["files"]["/proc/cpuinfo"] = f"ERROR: {exc}"
    else:
        info["files"]["/proc/cpuinfo"] = "not available"

    _fill_summary_from_lscpu(info)
    return info


def write_system_info(output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir) / "system_info.txt"
    info = collect_system_info()
    lines = ["# FlagGems Benchmark System Info", ""]

    lines.append("## Summary")
    for key, value in info["summary"].items():
        lines.append(f"{key}: {value}")
    lines.append("")

    for name, result in info["commands"].items():
        lines.append(f"## {name}")
        if result["returncode"] is None:
            lines.append(result["stderr"])
        else:
            if result["stdout"]:
                lines.append(result["stdout"].rstrip())
            if result["stderr"]:
                lines.append("")
                lines.append("[stderr]")
                lines.append(result["stderr"].rstrip())
            lines.append(f"[returncode] {result['returncode']}")
        lines.append("")

    for name, body in info["files"].items():
        lines.append(f"## {name}")
        lines.append(str(body).rstrip())
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return info


def _run_command(command: list[str]) -> dict[str, Any]:
    if shutil.which(command[0]) is None:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": f"{command[0]} not found in PATH",
        }
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": f"ERROR: {exc}"}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _fill_summary_from_lscpu(info: dict[str, Any]) -> None:
    lscpu_output = info["commands"].get("lscpu", {}).get("stdout") or ""
    for line in lscpu_output.splitlines():
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if key == "Model name":
            info["summary"]["cpu_model"] = value
        elif key == "CPU(s)":
            info["summary"]["cpu_count"] = value
        elif key == "NUMA node(s)":
            info["summary"]["numa_nodes"] = value

