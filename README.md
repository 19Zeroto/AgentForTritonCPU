# AgentForTritonCPU

Triton CPU 工作区的规则、任务 Spec、playbook、skill 和分析脚本仓库。产品源码
位于同一工作区的相邻 `triton-cpu/` 仓库，不复制到本仓。

## Layout

```text
AgentForTritonCPU/
├── AGENTS.md                 # 工作区入口
├── agent/
│   ├── AGENTS.md             # 任务路由与共享约束
│   ├── rules/                # 通用规则
│   ├── playbooks/            # 调试、测试、性能流程
│   ├── references/           # 项目地图、失败库、benchmark 约定
│   └── specs/tasks/          # 一次性任务契约
├── skills/
│   ├── environment/          # 环境加载与初始 bootstrap
│   ├── test-suite/            # 测试运行与结果汇总
│   ├── triton-cpu-rebuild/   # 通用重编译及可选 dgeglu 配方
│   ├── flaggems-benchmark-profile/
│   ├── flaggems-kernel-perf/
│   ├── support-matrix/
│   └── commit-bisect/
└── docs/                     # 面向人的完整阅读版文档
```

## Workspace

默认布局如下：

```text
$AGENT_DIR/
├── AgentForTritonCPU/
├── triton-cpu/
├── llvm-project/
├── logs/
└── cache/
```

`AGENT_DIR` 默认是 `$HOME/agent`；脚本会从自身位置推导
`AGENTFORTRITONCPU_DIR`，也可以显式覆盖 `TRITON_REPO_DIR`、`LLVM_INSTALL_DIR`、
`VENV_DIR`、`RUN_DIR` 和 `LOG_ROOT`。

加载共享环境：

```bash
source "$AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"
```

从 [docs/README.md](docs/README.md) 开始人类阅读；从
[agent/AGENTS.md](agent/AGENTS.md) 开始 Agent 任务路由。各
`skills/<name>/SKILL.md` 是对应工作流入口，详细执行约定放在其
`references/` 中。
日志、缓存和 benchmark 结果默认写入工作区 `logs/`、`cache/` 或 `/tmp`，不写入
本仓源码树。
