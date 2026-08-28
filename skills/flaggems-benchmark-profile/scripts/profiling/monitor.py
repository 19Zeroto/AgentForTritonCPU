from __future__ import annotations

import csv
import os
import threading
import time
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - depends on target environment
    psutil = None


class ResourceMonitor:
    """Background sampler for a process tree's CPU and memory usage."""

    def __init__(self, pid: int, interval_sec: float = 1.0):
        self.pid = pid
        self.interval_sec = interval_sec
        self.data: list[dict[str, float]] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_proc_ticks: int | None = None
        self._last_wall_time: float | None = None
        self._clock_ticks = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", "SC_CLK_TCK"))

    def start(self) -> None:
        self._prime_cpu_percent()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, float]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_sec * 2))
        return self.data

    def dump_csv(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["timestamp", "cpu_percent", "rss_mb", "vms_mb"],
            )
            writer.writeheader()
            writer.writerows(self.data)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            sample = self._sample_once()
            if sample is not None:
                self.data.append(sample)

    def _prime_cpu_percent(self) -> None:
        if psutil is None:
            self._last_proc_ticks = self._read_proc_tree_ticks()
            self._last_wall_time = time.time()
            return
        for process in self._psutil_process_tree():
            try:
                process.cpu_percent(interval=None)
            except Exception:
                continue

    def _sample_once(self) -> dict[str, float] | None:
        if psutil is None:
            return self._sample_procfs_once()

        processes = self._psutil_process_tree()
        if not processes:
            return None

        cpu_percent = 0.0
        rss_bytes = 0
        vms_bytes = 0
        for process in processes:
            try:
                with process.oneshot():
                    cpu_percent += float(process.cpu_percent(interval=None))
                    memory = process.memory_info()
                    rss_bytes += int(memory.rss)
                    vms_bytes += int(memory.vms)
            except Exception:
                continue

        return {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "rss_mb": rss_bytes / (1024 * 1024),
            "vms_mb": vms_bytes / (1024 * 1024),
        }

    def _psutil_process_tree(self) -> list[Any]:
        if psutil is None:
            return []
        try:
            root = psutil.Process(self.pid)
        except Exception:
            return []

        processes = [root]
        try:
            processes.extend(root.children(recursive=True))
        except Exception:
            pass
        return processes

    def _sample_procfs_once(self) -> dict[str, float] | None:
        pids = self._proc_tree_pids()
        if not pids:
            return None

        now = time.time()
        proc_ticks = 0
        rss_bytes = 0
        vms_bytes = 0
        for pid in pids:
            stat = self._read_proc_stat(pid)
            if stat is not None:
                proc_ticks += stat["utime"] + stat["stime"]
            memory = self._read_proc_statm(pid)
            if memory is not None:
                rss_bytes += memory["rss_bytes"]
                vms_bytes += memory["vms_bytes"]

        cpu_percent = 0.0
        if self._last_proc_ticks is not None and self._last_wall_time is not None:
            elapsed = now - self._last_wall_time
            if elapsed > 0:
                delta_ticks = proc_ticks - self._last_proc_ticks
                cpu_percent = delta_ticks / self._clock_ticks / elapsed * 100.0

        self._last_proc_ticks = proc_ticks
        self._last_wall_time = now
        return {
            "timestamp": now,
            "cpu_percent": cpu_percent,
            "rss_mb": rss_bytes / (1024 * 1024),
            "vms_mb": vms_bytes / (1024 * 1024),
        }

    def _read_proc_tree_ticks(self) -> int:
        total = 0
        for pid in self._proc_tree_pids():
            stat = self._read_proc_stat(pid)
            if stat is not None:
                total += stat["utime"] + stat["stime"]
        return total

    def _proc_tree_pids(self) -> list[int]:
        parent_to_children: dict[int, list[int]] = {}
        existing_pids = set()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            stat = self._read_proc_stat(pid)
            if stat is None:
                continue
            existing_pids.add(pid)
            parent_to_children.setdefault(stat["ppid"], []).append(pid)

        if self.pid not in existing_pids:
            return []

        result = []
        stack = [self.pid]
        while stack:
            pid = stack.pop()
            result.append(pid)
            stack.extend(parent_to_children.get(pid, []))
        return result

    @staticmethod
    def _read_proc_stat(pid: int) -> dict[str, int] | None:
        try:
            content = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            after_name = content.rsplit(")", 1)[1].split()
            return {
                "ppid": int(after_name[1]),
                "utime": int(after_name[11]),
                "stime": int(after_name[12]),
            }
        except (IndexError, ValueError):
            return None

    @staticmethod
    def _read_proc_statm(pid: int) -> dict[str, int] | None:
        try:
            parts = Path(f"/proc/{pid}/statm").read_text(encoding="utf-8").split()
        except OSError:
            return None
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            return {
                "vms_bytes": int(parts[0]) * page_size,
                "rss_bytes": int(parts[1]) * page_size,
            }
        except (IndexError, ValueError):
            return None
