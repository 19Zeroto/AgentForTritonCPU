# SiLU-and-mul 性能案例

这是一个历史实验的整理版，目的是帮助人类理解排查过程，不是当前默认配置，也
不是对所有 shape 的性能承诺。实验日期为 2026-06 至 2026-07，结论来自 Kunpeng
CPU backend 的多轮复测。

## 当前结论

- 连续的大输入通常会被 pointwise 调度压成 rank-1；决定性因素更多是 `numel`、
  tile 数量、dtype 和运行时内存状态，而不是原始 rank。
- f16 通常接近 Torch；f32 经常落后；bf16 最弱。单独调 tile size 不能稳定解决
  dtype 差异。
- `max_tile_size` 的收益不单调：128 在个别大 shape 上有收益，1024 在部分 f32
  场景有潜力，2048 经常回退。它们都不足以直接成为全局默认值。
- 小 shape 的启动和 wrapper 开销会掩盖真正的 kernel 差异；大 shape 才适合作为
  吞吐优化证据。
- perf 结果显示主要热点在生成的 SiLU-and-mul JIT kernel 及其 OpenMP 调用路径，
  不是 Python benchmark wrapper。

## 可复现实验边界

每次比较都应固定并记录：

- `test_silu_and_mul.py`、mode、level、shape source 和 dtype；
- warmup、iter、`OMP_NUM_THREADS`；
- 当前机器的 CPU/NUMA 绑定；
- Triton/LLVM revision、`TRITON_CACHE_DIR` 和 FlagGems cache；
- 是否包含 Torch reference、compile time 或 perf collector。

环境加载示例：

```bash
source "$AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"
```

建议先做单算子、单 dtype、单大 shape 的 dry-run，再进行多轮复测。所有输出写到
`$AGENT_DIR/logs` 或显式的外部目录，不写回源码树。

## 后续方向

1. 先用 compile log、benchmark record 和 perf 证据确认瓶颈；
2. 分别验证 tile、block pointer、1D layout 和 dtype lowering，避免一次改多个变量；
3. 每个候选方案做配对复测，并覆盖 f32、bf16 和代表性小 shape；
4. 只有在收益稳定且没有回归时，才考虑进入共享默认配置。

详细实验记录保存在 profiler skill 的历史 case study 中，Agent 使用时再按需加载。
