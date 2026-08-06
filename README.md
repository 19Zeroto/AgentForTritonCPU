# AgentForTritonCPU

Triton CPU 专用 Agent tooling/spec 仓。保存长期规则、任务 Spec、提示词、
测试与分析脚本；产品源码继续保存在相邻的 `triton-cpu` 仓。

## Layout

```text
agentfortritoncpu/
├── agent/
│   ├── AGENTS.md
│   ├── specs/
│   │   └── tasks/
│   ├── rules/
│   ├── playbooks/
│   ├── references/
│   └── prompts/
├── skills/
│   ├── test-suite/
│   ├── environment/
│   ├── commit-bisect/
│   ├── flaggems-benchmark-profile/
│   ├── support-matrix/
│   ├── silu-pointwise/
│   └── sme-benchmark/
└── docs/
```

## Workspace

默认目录关系：

```text
$AGENT_DIR/
├── agentfortritoncpu/
├── triton-cpu/
├── llvm-project/
├── logs/
└── cache/
```

`AGENT_DIR` 默认由脚本位置推导，也可显式设置。常用覆盖变量：
`TRITON_REPO_DIR`、`LLVM_INSTALL_DIR`、`VENV_DIR`、`RUN_DIR`、`LOG_ROOT`。

加载环境：

```bash
source "$AGENT_DIR/agentfortritoncpu/skills/environment/scripts/triton-cpu-env.sh"
```

每个 `skills/<name>/SKILL.md` 是 AI 统一入口；脚本在 `scripts/`，详细资料在
`references/`，UI 元数据在 `agents/openai.yaml`。测试入口见
[test-suite/SKILL.md](skills/test-suite/SKILL.md)。完整任务路由见
[agent/AGENTS.md](agent/AGENTS.md)。新任务 Spec 保存到
[`agent/specs/tasks/`](agent/specs/tasks/)。

这些 skill 仅供 Triton CPU 工作流使用，不安装为全局 skill，也不依赖 Codex
默认发现；`agent/AGENTS.md` 和相关 playbook 根据任务显式路由到对应入口。

该布局遵循 [Agent Skills specification](https://agentskills.io/specification)：
只在发现阶段读取 `SKILL.md` 元数据，触发后再加载工作流，并按需读取
`references/` 或执行 `scripts/`。

## Output policy

日志、缓存、状态文件、benchmark 结果不得默认写入本仓。脚本默认使用工作区
`logs/`、`cache/` 或 `/tmp`；需要保留自定义位置时显式传入输出目录。
