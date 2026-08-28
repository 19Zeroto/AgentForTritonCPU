# New Machine Setup Guide

新机器先安装 Codex，再克隆产品仓和本仓。本仓是 Triton CPU Agent 规则、
Spec、prompt、脚本的唯一来源；产品仓不保存这些文件或软链。

## Expected layout

```text
$AGENT_DIR/
├── AgentForTritonCPU/
├── triton-cpu/
├── llvm-project/
├── logs/
└── cache/
```

示例：

```bash
export AGENT_DIR="${AGENT_DIR:-$HOME/agent}"
git clone https://github.com/19Zeroto/AgentForTritonCPU.git \
  "$AGENT_DIR/AgentForTritonCPU"
```

`triton-cpu` 和 `llvm-project` 由各自流程准备。不要在存在未保存修改时替换仓目录。

## Workspace Agent entry

只在工作区根建立一个入口：

```bash
test ! -e "$AGENT_DIR/AGENTS.md"
ln -s AgentForTritonCPU/agent/AGENTS.md "$AGENT_DIR/AGENTS.md"
```

不得在 `triton-cpu` 内创建 `AGENTS.md`、`agents/` 或 Agent 脚本软链。确认：

```bash
test -L "$AGENT_DIR/AGENTS.md"
test "$(readlink "$AGENT_DIR/AGENTS.md")" = \
  "AgentForTritonCPU/agent/AGENTS.md"
test ! -e "$AGENT_DIR/triton-cpu/AGENTS.md"
test ! -e "$AGENT_DIR/triton-cpu/agents"
```

## Environment

```bash
export TRITON_REPO_DIR="$AGENT_DIR/triton-cpu"
export LLVM_INSTALL_DIR="$AGENT_DIR/llvm-project/install"
source "$AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"
```

脚本设计和调用入口位于 `skills/`。任务 Spec 保存到
`agent/specs/tasks/`。日志、cache、benchmark 结果保存到工作区 `logs/`、
`cache/` 或 `/tmp`，不得写入源码仓。

## Global Codex state

全局状态单独备份：

```text
~/.codex/AGENTS.md
~/.codex/config.toml
~/.codex/hooks.json
~/.codex/RTK.md
~/.codex/skills/        # 仅本地/私有 skill
```

默认排除 secrets、认证信息、session DB、插件 cache、安装包。公开工具和 skill
优先从上游重装。

## CodeGraph

`.codegraph/` 是机器本地生成索引，不迁移、不提交。新机器安装后重建：

```bash
codegraph init "$AGENT_DIR/triton-cpu"
codegraph status "$AGENT_DIR/triton-cpu"
```

## Smoke checks

```bash
codex --version
git -C "$AGENT_DIR/AgentForTritonCPU" status --short
git -C "$AGENT_DIR/triton-cpu" status --short
ls -l "$AGENT_DIR/AGENTS.md"
source "$AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"
```

大型 build、test、benchmark 前按 `agent/AGENTS.md` 加载对应规则。

