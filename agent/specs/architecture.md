# AgentForTritonCPU Architecture

## Repository roles

`AgentForTritonCPU` 保存 Agent 控制面；`triton-cpu` 保存产品代码；
`llvm-project` 保存 LLVM/MLIR 源码、构建树和安装树。三者是相邻目录，不通过产品仓
内的 Agent 文件、软链或脚本副本耦合。主要目标是通过 TritonShared + LLVM 支持
Kunpeng/AArch64 CPU，并适配、验证 FlagGems 算子。

工作区根 `AGENTS.md` 链接 `AgentForTritonCPU/AGENTS.md`；该文件再路由到
`AgentForTritonCPU/agent/AGENTS.md`。两级入口分别承担工作区路由和详细任务路由。

## Information layers

- `agent/specs/`：长期架构和任务 Spec。
- `agent/rules/`：所有任务必须遵守的约束。
- `agent/playbooks/`：按任务类型加载的执行流程。
- `agent/references/`：按需查询的项目地图、失败模式和历史证据。
- `skills/`：遵循 Agent Skills 格式、由本仓文档显式路由的工作流。每个 skill 使用
  `SKILL.md` 作为统一 AI 入口，`scripts/` 保存执行代码，`references/` 保存
  按需上下文，`agents/openai.yaml` 保存 UI 元数据。
- `docs/`：面向开发者和维护者的人类阅读版文档。它可以保留经过整理的内容副本，
  但不作为 Agent 执行规则或脚本契约的唯一来源。

任务先由 `agent/AGENTS.md` 路由，再加载最小必要上下文。正常调用脚本不读取实现
源码或无关 reference；修改脚本前按 `SKILL.md` 加载相应设计契约。新建和迁移
skill 统一遵守 `agent/rules/skill-authoring.md`。本仓不依赖 `.agents/skills`
或全局安装实现默认发现。

## Cross-cutting knowledge ownership

- 可执行的操作规则只在一个 rule 或 playbook 中定义；其他文档通过链接引用。
- 跨任务的技术概念只在一个共享 reference 中维护。
- Skill 保存入口、脚本和该专项独有的实验依据，不复制共享规则或背景知识。
- `agent/AGENTS.md` 只负责路由，不承载实现细节。
- 历史实验记录只保留仍能支持决策的证据，并标明日期和非规范性质；当前规则以
  所属 rule、playbook、维护中的 reference 和代码为准。

## Path contract

```text
AGENT_DIR=<workspace>
TRITON_REPO_DIR=$AGENT_DIR/triton-cpu
LLVM_INSTALL_DIR=$AGENT_DIR/llvm-project/install
AGENTFORTRITONCPU_DIR=<AgentForTritonCPU checkout>
```

脚本从自身位置推导 `AGENTFORTRITONCPU_DIR`；`AGENT_DIR` 默认是 `$HOME/agent`，
并可由环境变量覆盖。其他源码和安装路径默认相对 `AGENT_DIR` 解析，也必须允许
分别覆盖。日志、缓存、测试状态和报告默认保存于仓外。
