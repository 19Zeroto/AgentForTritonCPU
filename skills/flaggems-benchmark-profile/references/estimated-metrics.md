# Estimated metric fallback

FlagGems benchmark-local `get_tflops()` and `get_gbps()` implementations are
authoritative. `profiling/flops.py` is a presentation fallback for records that
lack native TFLOPS; it does not run inside the timed callable.

The aggregator records:

| Field | Meaning |
| --- | --- |
| `tflops_source=benchmark` | Native value from the benchmark record. |
| `tflops_source=estimated` | Static logical-FLOPs estimate divided by recorded latency. |
| `tflops_source=none` | Neither source is available. |
| `flops_formula_id` | Identifier for the selected static estimate. |

An estimate is a stable workload convention, not a hardware instruction count.
Special functions, comparisons, reductions, and fused pointwise operations do
not have a universal FLOPs definition. Never compare native and estimated
values without disclosing their sources.

Before adding or changing an estimate:

1. Confirm the current product benchmark does not already provide the metric.
2. Use only shape fields preserved in the benchmark record.
3. Give the formula a stable identifier and a short scope note.
4. Update `profiling/flops.py` and this contract together.
5. Test representative shapes, missing fields, and native-value precedence.

Inspect the live catalog rather than copying it into another document:

```bash
PYTHONPATH="$AGENT_DIR/AgentForTritonCPU/skills/flaggems-benchmark-profile/scripts" \
python3 -m profiling.flops --json
```
