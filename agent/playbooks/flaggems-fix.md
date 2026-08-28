---
name: flaggems-fix
description: 定位并修复 FlagGems 算子在 CPU/Kunpeng 路径上的正确性或编译问题
required_context:
  - agent/rules/core.md
  - agent/rules/script-writing.md
  - agent/rules/testing.md
  - agent/rules/reporting.md
  - agent/references/project-map.md
  - agent/references/failure-library.md
  - agent/playbooks/failure-debugging.md
  - agent/playbooks/testing.md
  - agent/playbooks/agent-handoff.md
---

# FlagGems 算子修复 Playbook

## 适用范围

用于 FlagGems 算子在 CPU/Kunpeng 路径上的测试失败、编译失败、运行时错误或数值不一致。根因属于共享 lowering 时，转入 `ir-lowering.md`。

## 主要入口

- 算子实现：`FlagGems/src/flag_gems/ops/`
- 融合算子：`FlagGems/src/flag_gems/fused/`
- 算子注册：`FlagGems/src/flag_gems/__init__.py`
- 自调优：`FlagGems/src/flag_gems/utils/libentry.py`
- Kunpeng 配置：`FlagGems/src/flag_gems/runtime/backend/_kunpeng/`
- 正确性测试：`FlagGems/tests/`

## 流程

1. 按 `failure-debugging.md` 提取首个根因，判定属于算子源码、backend 配置、runtime、lowering 还是测试期望。
2. 同时阅读目标算子、注册逻辑、backend 配置和失败用例。
3. 依次检查 GPU-only API 泄漏、CPU 不适用的 launch 参数、mask/索引越界、dtype promotion、axis/layout/broadcast 语义和 ARM/Kunpeng 配置。
4. 如果多个算子共享同一 op/pass 失败，或错误明确位于 bufferization、conversion、pointer analysis 或 LLVM codegen，停止算子层 workaround，转入 `ir-lowering.md`。
5. 只修改能解释当前失败的最小路径；不用容差、skip 或全局 JIT 改动掩盖根因。
6. 按 `testing.md` 验证原失败用例和可能受影响的 fallback 用例。

## 产出

- 目标算子、dtype、shape 和失败层级
- 根因证据与最小修复范围
- 修改文件及修改原因
- 已执行、未执行验证和期望成功信号
- 需要 lowering、环境或 review 任务时的转介信息
