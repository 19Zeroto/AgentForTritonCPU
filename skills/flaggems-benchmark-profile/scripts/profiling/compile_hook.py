from __future__ import annotations

import functools
import json
import os
import threading
import time
from typing import Any


_LOG_PATH = os.environ.get("FLAGGEMS_COMPILE_LOG")
_LOCK = threading.Lock()
_PENDING: dict[tuple[int, str], float] = {}


def install() -> None:
    if os.environ.get("FLAGGEMS_COMPILE_HOOK_DISABLE") == "1":
        return
    try:
        import triton
        from triton.runtime.jit import JITFunction
    except Exception:
        return

    if getattr(JITFunction, "_flaggems_profile_hook_installed", False):
        return

    if hasattr(JITFunction, "cache_hook") and hasattr(JITFunction, "compiled_hook"):
        _install_jitfunction_hooks(JITFunction)
    else:
        _install_jit_wrapper(triton)
    setattr(JITFunction, "_flaggems_profile_hook_installed", True)


def _install_jitfunction_hooks(jit_function_cls: Any) -> None:
    previous_cache_hook = getattr(jit_function_cls, "cache_hook", None)
    previous_compiled_hook = getattr(jit_function_cls, "compiled_hook", None)

    def cache_hook(**kwargs: Any) -> bool:
        skip_compile = False
        if previous_cache_hook is not None:
            skip_compile = bool(previous_cache_hook(**kwargs))
        if not skip_compile:
            _PENDING[_hook_key(kwargs)] = time.perf_counter()
        return skip_compile

    def compiled_hook(**kwargs: Any) -> Any:
        key = _hook_key(kwargs)
        start = _PENDING.pop(key, None)
        if start is not None:
            _write_record(
                _record_from_hook(kwargs, (time.perf_counter() - start) * 1000)
            )
        if previous_compiled_hook is not None:
            return previous_compiled_hook(**kwargs)
        return None

    jit_function_cls.cache_hook = cache_hook
    jit_function_cls.compiled_hook = compiled_hook


def _hook_key(kwargs: dict[str, Any]) -> tuple[int, str]:
    fn_info = kwargs.get("fn")
    jit_function = getattr(fn_info, "jit_function", None)
    compile_info = kwargs.get("compile") or {}
    return id(jit_function), str(compile_info.get("key", kwargs.get("key", "")))


def _record_from_hook(kwargs: dict[str, Any], compile_ms: float) -> dict[str, Any]:
    fn_info = kwargs.get("fn")
    compile_info = kwargs.get("compile") or {}
    constants = compile_info.get("constants") or {}
    signature = compile_info.get("signature") or {}
    name = getattr(fn_info, "name", None) or _name_from_repr(kwargs.get("repr"))
    return {
        "name": name,
        "module": getattr(fn_info, "module", None),
        "compile_ms": compile_ms,
        "signature": _json_safe(signature),
        "constants": _json_safe(constants),
        "repr": kwargs.get("repr"),
        "is_manual_warmup": bool(kwargs.get("is_manual_warmup", False)),
        "autotune": "autotune" in str(name).lower(),
    }


def _install_jit_wrapper(triton_module: Any) -> None:
    original_jit = triton_module.jit

    @functools.wraps(original_jit)
    def patched_jit(fn: Any = None, **kwargs: Any) -> Any:
        if fn is None:
            return lambda wrapped: _TimedJIT(original_jit(wrapped, **kwargs), wrapped)
        return _TimedJIT(original_jit(fn, **kwargs), fn)

    triton_module.jit = patched_jit


class _TimedJIT:
    def __init__(self, jit_function: Any, source_fn: Any):
        self._jit_function = jit_function
        self._source_fn = source_fn
        self._seen: set[str] = set()
        functools.update_wrapper(self, source_fn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._jit_function, name)

    def __getitem__(self, grid: Any) -> Any:
        launcher = self._jit_function[grid]

        @functools.wraps(launcher)
        def timed_launcher(*args: Any, **kwargs: Any) -> Any:
            return self._time_once("launch", grid, args, kwargs, launcher)

        return timed_launcher

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._time_once("call", None, args, kwargs, self._jit_function)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self._time_once(
            "run", kwargs.get("grid"), args, kwargs, self._jit_function.run
        )

    def _time_once(
        self,
        kind: str,
        grid: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
    ) -> Any:
        signature = _call_signature(args, kwargs)
        cache_key = json.dumps(signature, sort_keys=True, default=str)
        if cache_key in self._seen:
            return target(*args, **kwargs)
        start = time.perf_counter()
        result = target(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._seen.add(cache_key)
        _write_record(
            {
                "name": getattr(self._source_fn, "__name__", str(self._source_fn)),
                "module": getattr(self._source_fn, "__module__", None),
                "compile_ms": elapsed_ms,
                "signature": signature,
                "grid": _json_safe(grid),
                "kind": kind,
            }
        )
        return result


def _call_signature(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "args": [_describe_value(arg) for arg in args],
        "kwargs": {
            key: _describe_value(value) for key, value in sorted(kwargs.items())
        },
    }


def _describe_value(value: Any) -> Any:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "kind": type(value).__name__,
            "shape": list(getattr(value, "shape", [])),
            "dtype": str(getattr(value, "dtype", "")),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_describe_value(item) for item in value[:16]]
    if isinstance(value, dict):
        return {
            str(key): _describe_value(item)
            for key, item in list(value.items())[:32]
        }
    return type(value).__name__


def _write_record(record: dict[str, Any]) -> None:
    if not _LOG_PATH:
        return
    payload = json.dumps(record, default=str, sort_keys=True)
    with _LOCK:
        with open(_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(payload + "\n")


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, default=str)
        return value
    except TypeError:
        return str(value)


def _name_from_repr(repr_value: Any) -> str | None:
    if repr_value is None:
        return None
    text = str(repr_value)
    return text.split("[", 1)[0].split("(", 1)[0] or None


install()
