---
name: performance-analysis
description: 基于 FlagGems benchmark 结果识别编译、执行、tiling 或环境瓶颈
required_context:
  - agent/rules/core.md
  - agent/rules/script-writing.md
  - agent/rules/testing.md
  - agent/rules/reporting.md
  - agent/references/project-map.md
  - agent/references/benchmark-system.md
  - agent/references/failure-library.md
  - agent/playbooks/agent-handoff.md
---

# 性能分析 Playbook

## 适用范围

用于分析已有 FlagGems benchmark 结果中的 latency、speedup、TFLOPS、GB/s 和编译耗时。默认只分析和给出建议，不修改算子或编译器代码。

## 输入

- 完整 benchmark 命令和环境快照
- 目标模式、level、dtype、shape、warmup 和 iter 配置
- 原始结果日志，不只是汇总表
- PyTorch/reference 延迟与 FlagGems 延迟
- 编译耗时、backend 流水线状态和 OMP/affinity 配置

## 流程

1. 检查环境、输入、warmup/iter 和 reference 是否可比；不合格数据先标记，不直接下性能结论。
2. 按 `agent/references/benchmark-system.md` 确认 CPU 流水线及性能解释口径；
   再区分编译开销、operator 路径、wrapper 开销、流水线未生效、
   tiling/shape 不匹配和 OMP/NUMA 环境问题。
   `kernel` 模式是清缓存 `do_bench` 测试模式，不是单独 kernel 测量，
   不用于 kernel 热点分析；需要 kernel 级证据时使用 `perf` 或其他 profiling 工具。
3. 每个瓶颈结论必须引用具体指标、shape 或日志证据；区分已证实根因与待验证假设。
4. 按影响范围、回归风险和预期收益排序，不使用无数据支撑的 rough estimate。
5. 环境问题转入 `environment-setup.md`；算子配置或 kernel 问题转入 `flaggems-fix.md`；lowering/tiling 问题转入 `ir-lowering.md`。
6. 需要补采通用 benchmark、perf 或编译耗时时，读取
   `skills/flaggems-benchmark-profile/SKILL.md`。
7. 需要把 `perf.data` 归因到生成 kernel 的周期热点、子调用和进程外部开销，
   或输出指令级瓶颈报告时，读取 `skills/flaggems-kernel-perf/SKILL.md`。
8. 需要运行、重跑、导出或修改 fusion compute/memory 指标与公式时，读取
   `skills/flaggems-fusion-metrics/SKILL.md`。
9. 目标为 `silu_and_mul`/`mul` variant 时读取
   `skills/silu-pointwise/SKILL.md`；目标为 matmul、BLAS、baddbmm 或
   Stream-K 性能时读取 `skills/sme-benchmark/SKILL.md`。

## 产出

- 环境和数据可比性结论
- 按算子、dtype 和 shape 组织的关键指标
- 瓶颈层级、证据、已证实结论和假设
- 按优先级排序的后续任务
- 需要其他 Playbook 时的转介信息

Benchmark 标准执行命令尚未定义；本 Playbook 不提供或推测命令。
