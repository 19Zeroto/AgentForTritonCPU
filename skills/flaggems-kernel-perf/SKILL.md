---
name: flaggems-kernel-perf
description: Locate and report instruction- and call-tree-level bottlenecks in Triton CPU FlagGems generated kernels using per-shape perf recordings and period-weighted annotation. Use when asked which ARM instructions or child functions consume kernel cycles, or why kernel-inclusive time differs from whole-process time; not for correctness debugging or benchmark-only summaries.
---

# Analyze FlagGems Kernel Perf

Produce an evidence-backed report that separates whole-process hotspots, kernel
Self, kernel-inclusive Children, child symbols, and period-weighted instructions.

## Workflow

1. Load the `environment` skill. If `perf.data` does not already exist, also load
   `flaggems-benchmark-profile` and collect one process per shape in `operator`
   mode. Benchmark `kernel` mode is not isolated kernel measurement.
2. Read [references/workflow.md](references/workflow.md) before collecting or
   regenerating perf data. Keep every artifact under `$AGENT_DIR/logs`,
   `$AGENT_DIR/cache`, or an explicit external output directory.
3. Record at least two runs with identical event, frequency, call-graph mode,
   dtype, shape, benchmark arguments, affinity, NUMA policy, and cache policy.
   Use one profiling job. Never compare a no-callgraph run with an `fp` or
   `dwarf` run as if they were equivalent repeats.
4. Discover the exact generated kernel symbol from the flat perf report. Confirm
   any proposed child such as `memrefCopy` beneath that kernel in the call graph.
5. Analyze existing recordings with `scripts/analyze_perf.py`. Pass child symbols
   only after confirming their parentage:

   ```bash
   SKILL_DIR="${AGENT_DIR:-$HOME/agent}/AgentForTritonCPU/skills/flaggems-kernel-perf"
   python3 "$SKILL_DIR/scripts/analyze_perf.py" \
     "$RUN_A/perf.data" "$RUN_B/perf.data" \
     --kernel-symbol <exact-kernel-symbol> \
     --child-symbol <confirmed-child-symbol> \
     --shape '<shape>' \
     --output-dir "$OUT/analysis"
   ```

6. Read [references/interpretation.md](references/interpretation.md) before
   writing conclusions. Use `report.md`, `analysis.json`, CSV summaries, and raw
   reports as evidence; the generated report is a draft, not an automatic root
   cause verdict.

## Non-negotiable interpretation rules

- Use `perf annotate --show-total-period --percent-type local-period` results for
  instruction cost. Raw sample counts and static instruction counts are not
  instruction time shares.
- State every denominator: process Self, process-inclusive Children, or
  symbol-local period. Do not add Children and Self.
- Attribute an instruction to the symbol containing its address. Loads/stores in
  a child belong to the child's Self and the caller's inclusive tree, not the
  caller's Self.
- Treat `cycles` period as a sampled CPU-cycle/time proxy, not exact wall-clock
  duration. Do not claim memory bandwidth saturation from `ldr`/`str` share alone.
- `perf record` around pytest includes setup, input generation, JIT, Python, and
  runtime work. Compute outside-tree share only from kernel Children; decompose it
  only with call-chain evidence. Flat rows alone cannot identify the remainder.
- Missing permission, symbols, call chains, or counters are `N/A`, never zero.

## Acceptance

- Preserve `perf.data`, flat report, callgraph report, period annotation,
  commands, machine/binding context, and report for each shape.
- Require no lost samples or explain them; reject permission-denied,
  not-supported, and not-counted results.
- Compare repeated runs and identify configuration mismatch or unstable rankings.
- Report exact commands, event, shape, symbol, period/share, denominator, and the
  evidence file behind each conclusion.
