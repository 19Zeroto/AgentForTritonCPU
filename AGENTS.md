# Agent 总入口

## Workspace

本文件是 `$AGENT_DIR/AGENTS.md` 的唯一目标。工作区是多仓目录：

- `AgentForTritonCPU/`：Agent rules、specs、prompts、scripts。
- `triton-cpu/`：Triton CPU、TritonShared、FlagGems 产品源码。
- `llvm-project/`：LLVM/MLIR 源码与安装树。
- `logs/`、`cache/`：仓外运行产物。

使用 `AGENT_DIR` 表示工作区根目录，默认 `$HOME/agent`。仓操作必须
使用明确的 `git -C`，不得把工作区根目录当作单一 Git 仓。

## Routing

进入`triton-cpu`仓时，请先读取其对应的`AgentForTritonCPU/agent/AGENTS.md`