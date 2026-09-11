# FlagGems Fusion Metrics 实现说明

过去曾在 Agent skill 中维护一份独立的 fusion 公式和报告转换层。该层已经删除，
因为当前具体计算方法已经内嵌在主仓 FlagGems benchmark 的 `get_tflops()` 和
`get_gbps()` 中。主仓实现是唯一事实来源，本文只说明人类阅读和验证方法。

## 如何查找当前方法

进入相邻的 `triton-cpu/FlagGems/benchmark/` 仓库，找到目标 `test_*.py`：

1. 找到对应 benchmark 类；
2. 查看类的 `get_tflops()` 或 `get_gbps()` 实现；
3. 同时阅读该类的 `get_input_iter()`、`input_fn` 和 shape 配置，确认公式中的
   维度、dtype、mask、cache 或临时 buffer 与实际输入一致；
4. 以 benchmark record 中的 `tflops`/`gbps` 和 latency 作为运行结果，不从外部
   公式重新计算并覆盖它。

## 指标解释

`logical_flops` 和 `logical_bytes` 仍是用于横向比较的 logical metric，不等于硬件
指令数、真实 DRAM 流量或 roofline 上限。遇到 padding、tile、cache、临时 buffer
或硬件事务放大时，应区分 benchmark 方法表达的逻辑工作量与实现成本。

性能采集使用 [benchmark profile skill](../skills/flaggems-benchmark-profile/SKILL.md)，
kernel 热点使用 [kernel perf skill](../skills/flaggems-kernel-perf/SKILL.md)。
