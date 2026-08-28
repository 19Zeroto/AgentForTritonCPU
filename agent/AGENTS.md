# Triton CPU Agent

## Required context

- 任何代码修改：先读 `rules/core.md`。
- 编写、修改或运行脚本：同时读 `rules/script-writing.md`。
- 制作或修改 skill、迁移可复用脚本：同时读 `rules/skill-authoring.md`。
- 测试、复现、验证：读 `rules/testing.md` 和 `playbooks/testing.md`。
- TritonShared、MLIR、LLVM、transform、lowering：读 `rules/lowering.md`。
- Review：读 `rules/review.md`；完成报告读 `rules/reporting.md`。
- 架构、重构、需求澄清：读 `specs/architecture.md`、相关长期 Spec 和任务 Spec。
- Playbook frontmatter 声明的 `required_context` 必须完整加载。

## Task routing

| 任务 | Playbook | 查询资料 |
| --- | --- | --- |
| 测试与验证 | `playbooks/testing.md` | `references/project-map.md` |
| 失败定位 | `playbooks/failure-debugging.md` | `references/failure-library.md` |
| FlagGems 修复 | `playbooks/flaggems-fix.md` | `references/project-map.md` |
| IR/Lowering | `playbooks/ir-lowering.md` | `references/project-map.md` |
| 性能分析 | `playbooks/performance-analysis.md` | `references/benchmark-system.md` |
| 环境准备 | `playbooks/environment-setup.md` | `references/project-map.md` |
| 代码/方案审查 | `playbooks/code-review.md` | `references/failure-library.md` |
| Agent 转介 | `playbooks/agent-handoff.md` | 当前任务证据 |

## Skill routing

本仓 skill 不依赖 Codex 默认发现。任务命中下列条件时，先读取对应 `SKILL.md`，
再按该入口选择脚本和 reference：

| 任务条件 | Skill 入口 |
| --- | --- |
| 准备或检查 Triton CPU、LLVM/MLIR、FlagGems 环境 | `../skills/environment/SKILL.md` |
| 批量、并行、断点续跑或汇总 FlagGems correctness 测试 | `../skills/test-suite/SKILL.md` |
| 已有可复现 pytest 且已知 good/bad commit，需要定位首个回归 commit | `../skills/commit-bisect/SKILL.md` |
| 收集通用 FlagGems benchmark、perf、编译耗时、TFLOPS 或 profiling 报告 | `../skills/flaggems-benchmark-profile/SKILL.md` |
| 运行、导出或修改 FlagGems fusion compute/memory 指标与公式 | `../skills/flaggems-fusion-metrics/SKILL.md` |
| 生成或审计 Triton language、FlagGems operator 支持矩阵 | `../skills/support-matrix/SKILL.md` |

## Specs and scripts

- 新需求、执行方案、验收记录：`specs/tasks/`。
- 长期方向与架构：`specs/`。
- 强制约束：`rules/`。
- 可执行工具及设计：仓根 `skills/`。按 Skill routing 先读目标 skill 的
  `SKILL.md`；执行
  `scripts/` 中入口。只有修改脚本或 SKILL 指定时才读 `references/`。
- 新建 skill 或迁移脚本进入 `skills/`：按 `rules/skill-authoring.md` 完成归类、
  统一入口、路径修正和验证。
- 临时产物：`$AGENT_DIR/logs`、`$AGENT_DIR/cache` 或 `/tmp`，不得写回
  `triton-cpu` 和本仓源码目录。

禁止在未获用户明确授权时 commit 或 push。

## Compaction priority

1. 架构决策及理由
2. 修改文件与核心逻辑
3. 当前进展
4. 未完成 TODO 与验证结果
