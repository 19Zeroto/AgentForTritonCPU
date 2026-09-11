# Triton CPU Agent

## Required context

- 修改代码：读 `rules/core.md`；涉及脚本时再读 `rules/script-writing.md`。
- 制作或实质修改 skill：同时读 `rules/skill-authoring.md` 和内置
  `$skill-creator`。
- 测试、复现、验证：读 `rules/testing.md` 和 `playbooks/testing.md`。
- TritonShared、MLIR、LLVM、transform、lowering：读 `rules/lowering.md`。
- Review：读 `rules/review.md`；完成报告读 `rules/reporting.md`。
- 架构、重构、需求澄清：读 `specs/architecture.md` 和相关任务 Spec。
- Playbook frontmatter 声明的 `required_context` 必须完整加载。

## Task routing

| 任务 | Playbook | 按需资料 |
| --- | --- | --- |
| 定向测试与验证 | `playbooks/testing.md` | `references/project-map.md` |
| 失败定位 | `playbooks/failure-debugging.md` | `references/failure-library.md` |
| FlagGems 修复 | `playbooks/flaggems-fix.md` | `references/project-map.md` |
| IR/Lowering | `playbooks/ir-lowering.md` | `references/project-map.md` |
| 性能分析 | `playbooks/performance-analysis.md` | `references/benchmark-system.md` |
| 代码/方案审查 | `playbooks/code-review.md` | `references/failure-library.md` |

## Skill routing

本仓 skill 由此处显式路由。命中条件时先读对应 `SKILL.md`，再按入口选择脚本或
reference；不要默认加载整个 skill：

| 任务条件 | Skill 入口 |
| --- | --- |
| 准备或检查 Triton CPU、LLVM/MLIR、FlagGems 环境 | `../skills/environment/SKILL.md` |
| 更新 LLVM/MLIR、清理并重新安装 Triton CPU | `../skills/triton-cpu-rebuild/SKILL.md` |
| 重建后运行 dgeglu float32 验证 | `../skills/triton-cpu-rebuild/SKILL.md` 的 dgeglu recipe |
| 批量、并行、断点续跑或汇总 FlagGems correctness 测试 | `../skills/test-suite/SKILL.md` |
| 已有稳定 pytest 和 good/bad revision，定位首个回归 commit | `../skills/commit-bisect/SKILL.md` |
| 收集 FlagGems benchmark、编译耗时、资源或 perf 数据 | `../skills/flaggems-benchmark-profile/SKILL.md` |
| 归因生成 kernel 的周期热点、调用树和指令成本 | `../skills/flaggems-kernel-perf/SKILL.md` |
| 生成或审计 Triton language、FlagGems operator 支持矩阵 | `../skills/support-matrix/SKILL.md` |

## Ownership

- `rules/`：强制约束；`playbooks/`：任务决策流程；`references/`：跨任务事实。
- `specs/`：长期架构和任务范围、验收记录。
- `skills/`：专项入口、可执行工具和专项资料。正常执行脚本无需读取其实现源码。
- 历史材料必须声明日期和非规范性质；当前行为以代码、rules、playbooks 和维护中的
  reference 为准。
- 日志、缓存、状态和 benchmark 结果写入 `$AGENT_DIR/logs`、
  `$AGENT_DIR/cache` 或显式仓外目录。

禁止在未获用户明确授权时 commit 或 push。

## Compaction priority

1. 架构决策及理由
2. 修改文件与核心逻辑
3. 当前进展
4. 未完成 TODO 与验证结果
