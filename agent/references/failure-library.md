# 故障模式库

本文维护当前故障分类和修复边界。历史 MR 搜索入口见
`failure-history.md`；历史补丁不是当前行为的规范来源。

## 故障分类决策树

```
读取失败日志 → 找到首个根因错误
├── 报错提到 torch.cuda / device capability / accelerator → A. CPU-only适配问题
├── 报错 "Tensor-likes are not close" / AssertionError → B. 数值精度问题
├── 报错 segmentation fault / IndexError / OOB → C. 越界和段错误
├── 报错 bufferization / deallocation failed → D. Bufferization/内存生命周期
├── 报错 unexpected op / atomic / extern / Sleef / MakeTensorPtr → E. TritonShared/MLIR lowering
├── 报错在 test/benchmark 编译阶段 → F. FlagGems 测试/benchmark
├── 报错文件竞争 / 多进程 / pytest-xdist → G. 并发和调试
├── 报错编译太慢 / dispatch 太慢 → H. 性能问题
├── 上游同步后新出现的失败 → I. 上游同步/兼容性
```

## 快速映射表

| 故障信号 | 优先怀疑层级 | 常用修复方式 |
|---------|-------------|-------------|
| `torch.cpu` 缺少 CUDA 风格属性 | Python runtime / FlagGems backend | 替换为 CPU 属性或隔离 GPU-only 路径 |
| `Tensor-likes are not close` | FlagGems 算子 / dtype / mask / layout | 对齐 reference 语义，修正 axis/promotion/mask/rounding |
| segfault / OOB / wraparound | pointer analysis / indexing / layout | 修 mask gate、split pointer、wraparound、index bitwidth |
| bufferization/deallocation failed | MLIR bufferization / memory lifecycle | 调整 clone/dealloc/copy-before-write 顺序和 ownership |
| atomic / extern / Sleef / MakeTensorPtr lowering error | TritonShared lowering | 补齐 op lowering、类型分支或 pass 处理 |
| benchmark/test compile error | FlagGems test/benchmark harness | 修输入、注册、导入、skip list、kernel name |
| 多进程文件竞争 | test infrastructure / cache | 唯一文件、原子写入、明确 cache key |
| 编译或 dispatch 太慢 | compiler/driver/runtime performance | 缓存昂贵检查、调整 llc/LLVM/launcher/OMP 路径 |
| 上游同步后回归 | compatibility / backport / revert | 最小 backport 或定向 revert，保留 CPU backend 正确性 |

## 按层级分类的常见修复模式

### A. CPU-only 适配

**典型现象**：运行到 GPU-only API（`torch.cuda`、device capability、accelerator device、default generator）

**修复策略**：
1. 将 CUDA/HIP 设备查询替换为 CPU backend 可用属性或显式 fallback
2. Kunpeng/ARM 的 autotune/reduction/target feature 做专用配置，不污染通用 CPU 路径
3. 对缺少 CPU 实现但语义明确的算子提供 CPU fallback
4. 删除 CPU 不需要的 GPU launch 参数（`num_warps`、`num_stages`）

**避免误修**：
- 不要伪造 CUDA 设备对象
- 不要把 CPU backend 写成 "看起来像 GPU" 的兼容层
- 不要把 Kunpeng 专用配置放到所有 ARM 或所有 CPU 路径

### B. 数值精度

**典型现象**：fp16/bf16/fp32/fp8 结果偏差，mask 默认值、axis、layout、rounding 不一致

**修复策略**：
1. 对齐 PyTorch CPU 语义：axis、layout、broadcast、promotion、rounding、mask 默认值逐项核对
2. bf16/fp16 计算先升 fp32
3. masked load/store 明确 `other` 值
4. rounding 使用明确语义（RTNE 等）

**避免误修**：
- 不要把所有 `not close` 都当成容差问题
- 不要只改测试期望，除非能证明 CPU backend 语义是目标语义

### C. 越界和段错误

**典型现象**：segfault、IndexError、wraparound access、大 shape 下崩溃

**修复策略**：
1. load/store 和比较结果加 mask gate
2. 修正 1D gather、split pointer、wraparound 的边界推导
3. 对 index tensor、bitcast、pointer cast 使用正确 bitwidth
4. 大 shape 触发 LLVM crash 时检查 block size 和 reduction 维度

### D. Bufferization/内存生命周期

**典型现象**：bufferization/deallocation pass 失败，clone/copy-before-write/dealloc 顺序错误

**修复策略**：
1. 调整 clone lowering 与 buffer deallocation 顺序
2. 需要保留原始 buffer 语义时使用 copy-before-write
3. 修正 pass 中 iterator 生命周期
4. 保持 `scf.for` init value 与 region iter_arg 类型一致

### E. TritonShared/MLIR/LLVM lowering

**典型现象**：`unexpected op in ptr sequence`、bufferization failed、op 缺少 lowering

**修复策略**：
1. 为缺失 op 补齐 lowering 或类型分支
2. 修正 structured/unstructured pointer lowering 中的 pointer sequence/dynamic offset/split pointer 处理
3. 对 layout 非 row-major 问题在 conversion pass 内保持类型和 rank 一致
4. 对目标特征相关性能路径保持 backend pipeline 边界清晰

### F. FlagGems 测试/benchmark

**典型现象**：测试编译失败、benchmark 参数不适配、kernel name 语法错误

**修复策略**：
1. 修正输入生成、测试配置、导入、kernel name、skip list
2. 对 CPU 不适合 Triton kernel 的局部场景提供小范围 CPU fallback
3. 固定随机输入，让失败可复现

### G. 并发和调试

**典型现象**：pytest-xdist 文件竞争、缺少 per-kernel debug 信息

**修复策略**：
1. 多进程使用唯一文件、原子写入、明确 cache key
2. 增加 per-kernel debug 和 skip list 能力

### H. 性能

**典型现象**：编译时间过长、kernel dispatch overhead 过高

**修复策略**：
1. 降低 dispatch overhead：缓存昂贵检查结果
2. 编译性能：移除拖慢且无必要的 llc option
3. OMP 和目标特征优化保持 backend feature gate

### I. 上游同步/兼容性

**典型现象**：上游 Triton/FlagGems 变更引入 CPU backend 不兼容

**修复策略**：
1. Backport 只带入最小改动
2. Revert 时说明被破坏的 CPU 测试或 lowering 语义

## 任务路由

| 故障层级 | 主 Playbook | 可转入 |
|---------|-------------|---------|
| A. CPU-only适配 | `flaggems-fix.md` | — |
| B. 数值精度 | `flaggems-fix.md` | — |
| C. 越界/段错误 | `flaggems-fix.md` | `ir-lowering.md`（共享 lowering 问题） |
| D. Bufferization | `ir-lowering.md` | — |
| E. TritonShared lowering | `ir-lowering.md` | — |
| F. 测试/benchmark | `testing.md` / `performance-analysis.md` | `flaggems-fix.md`（算子失败） |
| G. 并发/调试 | `failure-debugging.md` | `skills/environment/SKILL.md` |
| H. 性能 | `performance-analysis.md` | `skills/environment/SKILL.md` / `ir-lowering.md` |
| I. 上游同步 | `flaggems-fix.md` | `ir-lowering.md`（共享 compiler 路径） |
