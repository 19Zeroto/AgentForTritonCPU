# Triton CPU 工作区入口

## Workspace

工作区根 `$AGENT_DIR/AGENTS.md` 应只链接到本文件。工作区是多仓目录：

- `AgentForTritonCPU/`：Agent rules、specs、prompts、scripts。
- `triton-cpu/`：Triton CPU、TritonShared、FlagGems 产品源码。
- `llvm-project/`：LLVM/MLIR 源码与安装树。
- `logs/`、`cache/`：仓外运行产物。

使用 `AGENT_DIR` 表示工作区根目录，默认 `$HOME/agent`。仓操作必须
使用明确的 `git -C`，不得把工作区根目录当作单一 Git 仓。

## Routing

处理 `AgentForTritonCPU/` 的规则、文档、playbook、reference、skill 或脚本，或进入
`triton-cpu/` 修改、测试和分析产品代码时，先完整读取
`AgentForTritonCPU/agent/AGENTS.md`。独立处理 `llvm-project/` 时遵循其仓内规则；
若工作同时涉及 Triton CPU lowering 或构建链路，也加载上述 Triton CPU 入口。
