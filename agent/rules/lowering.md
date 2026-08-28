# Transform 与 Lowering 规则

- 对 Triton Shared + LLVM 相关问题，先厘清 IR 流程：Triton 前端生成的 IR、Triton Shared 中间层、MLIR conversion、LLVM lowering、CPU codegen/runtime 调用之间的边界。
- 对 transform pipeline 修改，优先采用 transform-dialect 风格：emit 通用 transform sequence，让 matcher/PDL 决定是否命中 payload op。
- 特殊 lowering 路径应优先用显式 matcher 区分，不要在 regular matcher 里堆大量反向约束。
- 对 matcher 和 pipeline 代码，优先使用正向匹配，避免用大量 `!=` 约束隐藏特殊路径。
- 对当前 unsupported path，可使用显式 matcher 加 no-op action 保持 fallback 行为，避免被 regular path 误捕获。
- 不要将 f8、i8、f16、bf16 按位宽简单等价处理；每种类型都必须结合目标架构与 LLVM lowering 能力独立判断。
- `reduction_tile_size` 只能在已验证可 lower 的类型上用于将小位宽 reduction 对齐到 32-bit widening 需求。
- 对 bf16/f8 等 pending 支持，必须分开说明当前修复、未实现原因和后续可行路径。
- 修改 transform dialect 行为前，必须检查本地 LLVM verifier、TableGen 或 C++ 实现，确认 op 属性和约束合法，不得凭直觉修改。
