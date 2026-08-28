"""Auto-load the compile profiler when this directory is on PYTHONPATH."""

try:
    import compile_hook
except Exception:
    pass

