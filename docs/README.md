# Triton CPU 工作区文档

这里是面向开发者和维护者的人类阅读版文档。Agent 执行规则、任务路由、脚本
契约和机器可读资料分别位于 `agent/` 与 `skills/`，不需要为了阅读本目录而先
了解 Agent 内部结构。

## 文档目录

- [工作区设置](workspace-setup.md)：目录布局、入口软链和环境加载。
- [Benchmark shape 指南](benchmark_shape_guide_cn.md)：判断 shape 应该修改在哪一层。
- [Triton CPU 构建指南](build_triton_cpu.md)：首次准备 LLVM/MLIR、Python 和 Triton。
- [Fusion 指标实现说明](fusion_metrics_formula_derivation.md)：如何从主仓
  benchmark 的 `get_tflops()` / `get_gbps()` 阅读当前计算方法。
- [SiLU-and-mul 性能案例](silu_and_mul_perf_optimization_points.md)：历史实验的
  当前结论、可复现实验边界和后续方向。
- [旧工作区迁移说明](workspace_migration.md)：旧入口名称的兼容说明。

## 阅读原则

本目录的命令使用 `$AGENT_DIR`、环境变量和占位符，不绑定某台机器的 CPU、NUMA
节点或用户目录。性能数字和实验结论必须同时注明 shape、dtype、线程数、绑定、
工具链和缓存条件；历史案例不代表当前默认配置。
