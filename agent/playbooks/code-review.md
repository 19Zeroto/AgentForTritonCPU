---
name: code-review
description: 审查 triton-cpu 代码或设计变更，识别正确性、回归、验证和范围问题
required_context:
  - agent/rules/core.md
  - agent/rules/script-writing.md
  - agent/rules/review.md
  - agent/rules/testing.md
  - agent/rules/reporting.md
  - agent/references/project-map.md
  - agent/references/failure-library.md
---

# 代码审查 Playbook

## 适用范围

用于审查代码 diff、方案设计或修复结果。默认只报告问题和建议；只有用户明确要求修改时才实施修复。

## 流程

1. 确认审查范围、当前分支、工作区、未跟踪文件和相关 commit。
2. 逐文件阅读 diff 及相关上下文，确认变更意图、失败路径和预期验证。
3. 优先检查可证明的正确性、安全性和回归问题，再检查设计权衡和非阻塞细节。
4. 对 FlagGems 变更，检查 CPU/GPU API 边界、dtype、axis/layout/mask、launch 参数、backend fallback 范围和目标特征验证。
5. 对 compiler/lowering 变更，检查 pass 归属、rewrite 类型与 ownership、feature gate、matcher/fallback 覆盖和默认语义。
6. 对性能变更，检查环境可比性、原始指标、预热/重复次数、目标 shape 和非目标路径回归风险。
7. 核对已执行测试的实际覆盖范围；不把 targeted case 通过当作全量验证。

## 输出

每个可操作问题至少包含：

- 优先级和是否阻塞
- 文件与精确代码位置
- 具体问题和可触发场景
- 为什么当前行为错误或存在回归风险
- 最小修正方向或需要补充的验证

无可操作问题时，明确说明未发现问题，并列出残余风险和未覆盖测试。范围外问题可记录为非阻塞后续项，不扩大当前 review。
