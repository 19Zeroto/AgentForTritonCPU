from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VALID_MODES = {"kernel", "operator", "wrapper"}
VALID_LEVELS = {"core", "comprehensive"}
DEFAULT_CACHE_EVENTS = [
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "L1-dcache-stores",
    "LLC-loads",
    "LLC-load-misses",
    "LLC-stores",
    "dTLB-load-misses",
    "branch-misses",
    "instructions",
    "cycles",
]


@dataclass
class SystemConfig:
    triton_cache_dir: str | None = None
    clear_triton_cache: bool = True
    numa_enabled: bool = False
    numa_cpunodebind: int | None = None
    numa_membind: int | None = None
    cpu_affinity: str | None = None
    env_vars: dict[str, str] = field(default_factory=dict)


@dataclass
class ProfilingConfig:
    cpu_memory_enabled: bool = True
    cpu_memory_interval_sec: float = 1.0
    cache_metrics_enabled: bool = True
    cache_backend: str = "perf"
    cache_events: list[str] = field(default_factory=lambda: DEFAULT_CACHE_EVENTS[:])
    compilation_time_enabled: bool = True


@dataclass
class TierConfig:
    warmup: int
    iterations: int


@dataclass
class TestEntry:
    test_file: str
    marker: str
    tier: str = "medium"
    mode: str = "kernel"
    level: str = "core"
    warmup_override: int | None = None
    iter_override: int | None = None
    metrics: list[str] | None = None
    dtypes: list[str] | None = None


@dataclass
class RunConfig:
    tiers: dict[str, TierConfig]
    global_mode: str
    global_level: str
    default_tier: str
    system: SystemConfig
    profiling: ProfilingConfig
    tests: list[TestEntry]
    config_path: str
    benchmark_dir: str
    global_metrics: list[str] | None = None
    global_dtypes: list[str] | None = None

    @classmethod
    def from_yaml(
        cls, path: str | Path, benchmark_dir: str | Path | None = None
    ) -> "RunConfig":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to load profiling_config.yaml. "
                "Install it in the target benchmark environment first."
            ) from exc

        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Profiling config must be a mapping: {config_path}")

        resolved_benchmark_dir = (
            Path(benchmark_dir).resolve()
            if benchmark_dir is not None
            else config_path.parent
        )
        tiers = _parse_tiers(payload.get("tiers"))
        global_payload = payload.get("global") or {}
        if not isinstance(global_payload, dict):
            raise ValueError("'global' section must be a mapping")

        global_mode = str(global_payload.get("mode", "kernel"))
        global_level = str(global_payload.get("level", "core"))
        default_tier = str(global_payload.get("default_tier", "medium"))
        global_metrics = _optional_str_list(
            global_payload.get("metrics"), "global.metrics"
        )
        global_dtypes = _optional_str_list(
            global_payload.get("dtypes"), "global.dtypes"
        )

        _validate_choice("global.mode", global_mode, VALID_MODES)
        _validate_choice("global.level", global_level, VALID_LEVELS)
        if default_tier not in tiers:
            raise ValueError(
                f"global.default_tier '{default_tier}' is not defined in tiers"
            )

        system = _parse_system_config(payload.get("system") or {})
        profiling = _parse_profiling_config(payload.get("profiling") or {})
        tests = _parse_tests(
            payload.get("tests"),
            tiers,
            default_tier,
            global_mode,
            global_level,
            global_metrics,
            global_dtypes,
            resolved_benchmark_dir,
        )

        return cls(
            tiers=tiers,
            global_mode=global_mode,
            global_level=global_level,
            default_tier=default_tier,
            system=system,
            profiling=profiling,
            tests=tests,
            config_path=str(config_path),
            benchmark_dir=str(resolved_benchmark_dir),
            global_metrics=global_metrics,
            global_dtypes=global_dtypes,
        )


def _parse_tiers(raw: Any) -> dict[str, TierConfig]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("'tiers' section must define at least one tier")
    tiers: dict[str, TierConfig] = {}
    for name, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"tier '{name}' must be a mapping")
        warmup = _positive_int(value.get("warmup"), f"tiers.{name}.warmup")
        iterations = _positive_int(value.get("iterations"), f"tiers.{name}.iterations")
        tiers[str(name)] = TierConfig(warmup=warmup, iterations=iterations)
    return tiers


def _parse_system_config(raw: Any) -> SystemConfig:
    if not isinstance(raw, dict):
        raise ValueError("'system' section must be a mapping")

    numa_raw = raw.get("numa") or {}
    if not isinstance(numa_raw, dict):
        raise ValueError("system.numa must be a mapping")
    cpunodebind = _optional_int(numa_raw.get("cpunodebind"), "system.numa.cpunodebind")
    membind = _optional_int(numa_raw.get("membind"), "system.numa.membind")
    numa_enabled = bool(
        numa_raw.get(
            "enabled",
            cpunodebind is not None or membind is not None,
        )
    )

    env_vars_raw = raw.get("env_vars") or {}
    if not isinstance(env_vars_raw, dict):
        raise ValueError("system.env_vars must be a mapping")
    env_vars = {str(key): str(value) for key, value in env_vars_raw.items()}

    triton_cache_dir = raw.get("triton_cache_dir")
    return SystemConfig(
        triton_cache_dir=str(triton_cache_dir) if triton_cache_dir else None,
        clear_triton_cache=bool(raw.get("clear_triton_cache", True)),
        numa_enabled=numa_enabled,
        numa_cpunodebind=cpunodebind,
        numa_membind=membind,
        cpu_affinity=str(raw["cpu_affinity"]) if raw.get("cpu_affinity") else None,
        env_vars=env_vars,
    )


def _parse_profiling_config(raw: Any) -> ProfilingConfig:
    if not isinstance(raw, dict):
        raise ValueError("'profiling' section must be a mapping")

    cpu_memory = raw.get("cpu_memory") or {}
    cache_metrics = raw.get("cache_metrics") or {}
    compilation_time = raw.get("compilation_time") or {}
    for section_name, section in (
        ("profiling.cpu_memory", cpu_memory),
        ("profiling.cache_metrics", cache_metrics),
        ("profiling.compilation_time", compilation_time),
    ):
        if not isinstance(section, dict):
            raise ValueError(f"{section_name} must be a mapping")

    events = _optional_str_list(
        cache_metrics.get("events"), "profiling.cache_metrics.events"
    )
    return ProfilingConfig(
        cpu_memory_enabled=bool(cpu_memory.get("enabled", True)),
        cpu_memory_interval_sec=_positive_float(
            cpu_memory.get("interval_sec", 1.0), "profiling.cpu_memory.interval_sec"
        ),
        cache_metrics_enabled=bool(cache_metrics.get("enabled", True)),
        cache_backend=str(cache_metrics.get("backend", "perf")),
        cache_events=events if events is not None else DEFAULT_CACHE_EVENTS[:],
        compilation_time_enabled=bool(compilation_time.get("enabled", True)),
    )


def _parse_tests(
    raw: Any,
    tiers: dict[str, TierConfig],
    default_tier: str,
    global_mode: str,
    global_level: str,
    global_metrics: list[str] | None,
    global_dtypes: list[str] | None,
    benchmark_dir: Path,
) -> list[TestEntry]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("'tests' section must define at least one test")

    tests: list[TestEntry] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"tests[{index}] must be a mapping")
        test_file = item.get("test_file")
        marker = item.get("marker")
        if not test_file or not marker:
            raise ValueError(f"tests[{index}] must define test_file and marker")

        tier = str(item.get("tier", default_tier))
        if tier not in tiers:
            raise ValueError(f"tests[{index}].tier '{tier}' is not defined in tiers")

        mode = str(item.get("mode", global_mode))
        level = str(item.get("level", global_level))
        _validate_choice(f"tests[{index}].mode", mode, VALID_MODES)
        _validate_choice(f"tests[{index}].level", level, VALID_LEVELS)

        test_path = Path(str(test_file))
        if not test_path.is_absolute():
            test_path = benchmark_dir / test_path
        if not test_path.exists():
            raise FileNotFoundError(f"benchmark test file does not exist: {test_path}")

        tests.append(
            TestEntry(
                test_file=str(test_file),
                marker=str(marker),
                tier=tier,
                mode=mode,
                level=level,
                warmup_override=_optional_positive_int(
                    item.get("warmup_override"), f"tests[{index}].warmup_override"
                ),
                iter_override=_optional_positive_int(
                    item.get("iter_override"), f"tests[{index}].iter_override"
                ),
                metrics=_optional_str_list(
                    item.get("metrics"), f"tests[{index}].metrics"
                )
                if "metrics" in item
                else global_metrics,
                dtypes=_optional_str_list(
                    item.get("dtypes"), f"tests[{index}].dtypes"
                )
                if "dtypes" in item
                else global_dtypes,
            )
        )
    return tests


def _positive_int(value: Any, name: str) -> int:
    parsed = _optional_int(value, name)
    if parsed is None or parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _optional_positive_int(value: Any, name: str) -> int | None:
    parsed = _optional_int(value, name)
    if parsed is not None and parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _optional_str_list(value: Any, name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [str(item) for item in value]


def _validate_choice(name: str, value: str, choices: set[str]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}")
