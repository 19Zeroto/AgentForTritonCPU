# Workspace setup

这是一份面向开发者的人类阅读版设置说明；Agent 执行时使用
`skills/environment/SKILL.md` 和对应脚本。

## Layout

```text
$AGENT_DIR/
├── AGENTS.md -> AgentForTritonCPU/AGENTS.md
├── AgentForTritonCPU/
├── triton-cpu/
├── llvm-project/
├── logs/
└── cache/
```

`AGENT_DIR` defaults to `$HOME/agent`. Clone the repositories into that layout
or set `AGENT_DIR`, `TRITON_REPO_DIR`, and `LLVM_INSTALL_DIR` explicitly.

## Agent entry

Create one workspace-level link to the lightweight multi-repository router:

```bash
AGENT_DIR="${AGENT_DIR:-$HOME/agent}"
test ! -e "$AGENT_DIR/AGENTS.md"
ln -s AgentForTritonCPU/AGENTS.md "$AGENT_DIR/AGENTS.md"
```

Do not add Agent instructions or script links to `triton-cpu`. Verify:

```bash
test -L "$AGENT_DIR/AGENTS.md"
test "$(readlink "$AGENT_DIR/AGENTS.md")" = 'AgentForTritonCPU/AGENTS.md'
test ! -e "$AGENT_DIR/triton-cpu/AGENTS.md"
```

The linked router sends Triton CPU and Agent control-plane work to
`AgentForTritonCPU/agent/AGENTS.md`.

## Toolchain

For a new LLVM/Triton installation, read
`skills/environment/references/bootstrap.md`. For routine activation:

```bash
source "$AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"
```

Keep logs, caches, state, benchmark results, and generated support matrices out
of both source repositories.

## Local-only state

Codex configuration, credentials, sessions, plugin caches, and generated code
indexes are machine-local state. Reinstall public skills/plugins from their
source and back up only intentionally selected configuration. Never publish
secrets or authentication material with this repository.

## Smoke checks

```bash
git -C "$AGENT_DIR/AgentForTritonCPU" status --short
git -C "$AGENT_DIR/triton-cpu" status --short
readlink "$AGENT_DIR/AGENTS.md"
source "$AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"
```
