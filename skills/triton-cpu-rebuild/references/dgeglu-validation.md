# dgeglu rebuild validation recipe

This recipe composes `scripts/rebuild.sh` with the main FlagGems dgeglu
benchmark. It is an acceptance recipe for work where `dgeglu` float32 is
relevant, not the default validation for every compiler change.

The benchmark uses `operator` mode, `core` shapes, float32, five warmups, and
five measured iterations. CPU and memory NUMA nodes are required arguments;
the CPU list may be supplied explicitly or selected from the chosen CPU node.
All binding values must describe the current machine.

Use an outer `numactl`/`taskset` only when the rebuild and benchmark must share
one scheduling policy. In that case, keep the outer policy consistent with the
recipe arguments and `OMP_NUM_THREADS`.

The benchmark's local `get_gbps()` implementation is the metric source of
truth. The recipe invokes `triton-cpu/FlagGems/benchmark/test_dgeglu.py`
directly, keeps record logs in an external run directory, and does not
duplicate formulas or interpret a successful benchmark as full correctness
coverage.
