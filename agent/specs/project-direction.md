# 项目长期方向

本仓库是 `triton-cpu` 开发工作区。后续 agent 进入仓库时，应优先围绕 CPU 后端适配展开，而不是把它当作普通上游 Triton 仓库处理。

当前主要开发方向：

- 将 Triton 通过 Triton Shared + LLVM 的方式连接到 CPU 上执行。
- 主要后端机器是 Kunpeng CPU，涉及 AArch64/ARM 侧的编译、运行时和性能问题时需要优先考虑 Kunpeng 环境。
- `FlagGems/` 中包含一批 Triton 算子，这些算子需要做 CPU 端适配、编译链路打通和行为验证。
