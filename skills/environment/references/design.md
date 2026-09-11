# Triton CPU Environment Helper

入口：`scripts/triton-cpu-env.sh`。必须 `source`，用于激活 Python 环境并导出
LLVM/MLIR、TritonShared、FlagGems 所需变量。`AGENT_DIR` 是工作区/安装根目录，默认
为 `$HOME/agent`，并优先接受调用者覆盖；其他路径默认相对 `AGENT_DIR` 解析，也可
分别通过环境变量覆盖。详细构建说明见 `../../docs/build_triton_cpu.md`。
