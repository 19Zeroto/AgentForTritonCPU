# AgentForTritonCPU Architecture

## Repository roles

`AgentForTritonCPU` 保存 Agent 控制面；`triton-cpu` 保存产品代码。两者为相邻仓，
不通过产品仓内软链或脚本副本耦合。工作区根 `AGENTS.md` 负责发现本仓入口。

## Information layers

- `agent/specs/`：长期目标、架构、任务 Spec。
- `agent/rules/`：所有任务必须遵守的约束。
- `agent/playbooks/`：按任务类型加载的执行流程。
- `agent/references/`：按需查询的项目地图、失败模式和历史证据。
- `agent/prompts/`：可复用提示模板。
- `skills/`：遵循 Agent Skills 格式、由本仓文档显式路由的工作流。每个 skill 使用
  `SKILL.md` 作为统一 AI 入口，`scripts/` 保存执行代码，`references/` 保存
  按需上下文，`agents/openai.yaml` 保存 UI 元数据。

任务先由 `agent/AGENTS.md` 路由，再加载最小必要上下文。正常调用脚本不读取长篇
设计资料；修改脚本前按 `SKILL.md` 路由读取相关 reference。新建和迁移流程统一
遵守 `agent/rules/skill-authoring.md`。本仓不依赖 `.agents/skills` 或全局安装
实现默认发现。

## Cross-cutting knowledge ownership

- 可执行的操作规则只在一个 rule 或 playbook 中定义；其他文档通过链接引用。
- 跨任务的技术概念只在一个共享 reference 中维护。
- Skill 保存入口、脚本和该专项独有的实验依据，不复制共享规则或背景知识。
- `agent/AGENTS.md` 只负责路由，不承载实现细节。
- 历史实验记录必须标明其非规范性质；当前规则以所属 rule、playbook 或共享
  reference 为准。

## Path contract

```text
AGENT_DIR=<workspace>
TRITON_REPO_DIR=$AGENT_DIR/triton-cpu
LLVM_INSTALL_DIR=$AGENT_DIR/llvm-project/install
AGENTFORTRITONCPU_DIR=$AGENT_DIR/AgentForTritonCPU
```

脚本优先从自身位置推导 `AGENTFORTRITONCPU_DIR` 和 `AGENT_DIR`，环境变量可覆盖。
日志、缓存、测试状态默认保存于仓外。
