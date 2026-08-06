# Triton CPU Environment Helper

入口：`scripts/triton-cpu-env.sh`。必须 `source`，用于激活 Python 环境并导出
LLVM/MLIR、TritonShared、FlagGems 所需变量。路径优先接受调用者覆盖；默认从本
skill 位置反推工作区。详细构建说明见 `../../docs/build_triton_cpu.md`。
