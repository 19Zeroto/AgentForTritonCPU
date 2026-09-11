---
name: ir-lowering
description: 定位并修复 TritonShared、MLIR 和 LLVM lowering 路径问题
required_context:
  - agent/rules/core.md
  - agent/rules/script-writing.md
  - agent/rules/testing.md
  - agent/rules/lowering.md
  - agent/rules/reporting.md
  - agent/references/project-map.md
  - agent/references/failure-library.md
  - agent/playbooks/failure-debugging.md
  - agent/playbooks/testing.md
---

# IR/Lowering 修复 Playbook

## 适用范围

用于 TritonShared conversion、bufferization、pointer/mask analysis、transform dialect、vector/LLVM lowering 和 LLVM codegen 问题。

## 主要路径

- 流水线入口：`triton-shared/backend/compiler.py`
- Conversion：`triton-shared/lib/Conversion/`
- Analysis：`triton-shared/lib/Analysis/` 和 `AnalysisStructured/`
- Dialect：`triton-shared/lib/Dialect/`
- 独立调试工具：`triton-shared/tools/triton-shared-opt/`
- Lit 测试：`triton-shared/test/`

## 流程

1. 按 `failure-debugging.md` 确认首个失败 op、pass、dtype、shape 和流水线。
2. 使用 `TRITON_SHARED_DUMP_PATH` 获取失败边界前后的 IR，定位首个破坏语义或 verifier 约束的 pass。
3. 确认 op 属于哪个 dialect、应在哪个 conversion 阶段处理，以及 rewrite 前后的类型、rank、layout 和 ownership 约束。
4. 优先修改真正缺失或错误的 pattern/analysis；不通过禁用 pass、跳过 verifier 或在算子层复制大段 buffer 规避问题。
5. 使用本地 verifier、TableGen/C++ 实现和 `triton-shared-opt` 验证独立 IR 变换。
6. 验证原失败用例、可能被 matcher 误捕获的 fallback 用例，以及必要的 Lit 或 conversion 测试。

## 调试入口

```bash
TRITON_SHARED_DUMP_PATH=/tmp/mlir_dump \
OMP_NUM_THREADS=32 \
<reproduction-command>

triton-shared-opt input.mlir --pass-pipeline="<pipeline>" -o output.mlir

MLIR_ENABLE_DUMP=1 triton-shared-opt input.mlir --mlir-print-ir-after-all
```

## 产出

- 失败 pass/op 和首个错误 IR
- 根因所属 conversion/analysis 边界
- 修改前后的关键 IR 差异及语义理由
- 已执行与未执行验证
- 对 FlagGems 算子层、环境或 review 任务的转介信息
