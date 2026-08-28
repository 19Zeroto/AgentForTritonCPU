# AI 修复方法论：从 master 已合入 PR 归纳

本文归纳 `master` 已合入 `!NN` PR/MR 中反复出现的问题类型和修复方式，用作后续 AI 修复 `triton-cpu`、TritonShared 和 FlagGems CPU 适配问题时的参考。重点不是复述每个 PR 的完整 diff，而是总结可复用的判断路径：先识别问题属于哪一层，再匹配已有修复模式，最后做最小范围改动。

## 总体原则

1. 先定位首个根因错误，不要只看最后一行失败信息。
2. 先判定层级：测试输入、FlagGems 算子源码、Python runtime/JIT、TritonShared lowering、MLIR/LLVM codegen、runtime/driver、backend 配置。
3. 优先复用历史 PR 的修复模式，不把算子问题扩大成 compiler 重构，也不用测试容差掩盖真实越界或 lowering 错误。
4. 只修能解释当前失败的最小路径；CPU/Kunpeng 适配优先保持 backend 隔离，避免把 GPU-only 假设继续扩散。
5. 每次修复都要给出目标验证点：失败用例、失败 op/pass、关键日志片段、benchmark 指标或确定性的复现输入。

## CPU-only 适配问题

### 典型现象

- 运行到 `torch.cuda`、device capability、accelerator device、default generator 等 GPU-only API。
- CPU backend 被迫读取 GPU metadata，例如 `multi_processor_count`、`get_device_properties`、`get_device_capability`。
- Kunpeng/ARM 配置混在通用 CPU 配置里，导致不适用的 `num_warps`、`num_stages`、reduction 或 target feature 被使用。

### 根因判断

优先检查失败调用是否来自 Python runtime、FlagGems backend dispatch、vendor config 或测试工具，而不是直接进入 MLIR/LLVM 层。若错误文本指向 `torch.cpu` 缺少某个 CUDA 风格属性，根因通常是 GPU backend 假设泄漏到 CPU 路径。

### 修复策略

- 将 CUDA/HIP 设备查询替换成 CPU backend 可用的属性、常量或显式 fallback。
- 对 Kunpeng/ARM 的 autotune、reduction、target feature 做专用配置，避免污染通用 CPU 路径。
- 对缺少 CPU 实现但语义明确的算子，提供 CPU fallback；fallback 要局限在对应算子或 backend 分支。
- 删除 CPU 不需要、且会触发错误的 GPU launch 参数，例如 `num_warps`、`num_stages`。

### 避免误修

- 不要为了绕过错误伪造 CUDA 设备对象。
- 不要把 CPU backend 写成“看起来像 GPU”的兼容层，除非上层接口确实要求。
- 不要把 Kunpeng 专用配置放到所有 ARM 或所有 CPU 路径。

### 代表 PR

`!26` CPU device random generator/task partition，`!27` AttributeError，`!32` default generators，`!33` get_device_capability，`!34`/`!113` get_device_properties，`!37` CPU only，`!39` accelerator device，`!42` multi_processor_count，`!49`/`!56` Torch not compiled with CUDA enabled，`!81` missing `USE_INT32_IDX` for arm，`!111` no triton-shared backend scenario，`!112` Kunpeng `naive_reduction` config，`!155` remove `num_stages`/`num_warps`，`!162` FlagGems backend test。

## 数值精度不一致问题

### 典型现象

- 测试报 `Tensor-likes are not close`、`AssertionError`、scalar/tensor not close。
- fp16/bf16/fp32、fp8、integer index 或 type promotion 后结果偏差。
- mask 默认值、axis、layout、rounding、reduction 顺序与 PyTorch/CPU reference 不一致。

### 根因判断

先确认预期语义来自 PyTorch reference、FlagGems 算子定义还是 Triton lowering。若错误集中在某个算子或 dtype，优先看算子源码和 dtype promotion；若多个算子共享同一 op 或 lowering 失败，再下探到 TritonShared/LLVM。

### 修复策略

- 对齐 PyTorch CPU 语义：axis、layout、broadcast、promotion、rounding、mask 默认值要逐项核对。
- 对低精度计算使用必要的中间类型提升，例如 bf16/fp16 先升 fp32 再计算。
- 对 masked load/store 明确 `other` 值，避免未定义值参与结果。
- 对 rounding 使用明确语义，例如 RTNE、C `%` 语义或 `RoundEven` 映射。
- 对 reduction 类算子优先修 block size、mask 和 reduction 维度，不先放宽 tolerance。

### 避免误修

- 不要把所有 `not close` 都当成容差问题；很多历史修复来自错误 axis、mask、layout 或 dtype。
- 不要只改测试期望，除非能证明 CPU backend 的语义本身就是目标语义。
- 不要在算子层补偿 compiler 层的确定性 lowering 错误。

### 代表 PR

`!31`/`!54`/`!60`/`!70` pointwise not close，`!35` float layout，`!44` pointwise_dynamic and quant，`!58` type promotion，`!67` log_softmax backward masked load，`!69` float16 `fexpand`，`!82` quant，`!83` GELU bf16 promote，`!96` fp8 RTNE rounding，`!97` softmax backward default `other`，`!119` weightnorm，`!122` sparse MLA，`!123` batch norm backward，`!127` vector norm，`!129` attention，`!130` nll_loss，`!132` layernorm backward，`!145` reshape and cache，`!154` multinomial。

## 越界访问和段错误问题

### 典型现象

- 运行时 segmentation fault。
- IndexError、wraparound access、1D gather/split pointer、masked load 后出现异常。
- LLVM backend crash 只在大 shape 或特定 layout 下出现。

### 根因判断

优先检查索引表达式、mask 条件、指针重写、split pointer、wraparound、layout/stride 和 index bitwidth。若问题与 `StructuredToMemref`、PtrAnalysis 或 unstructured ptr lowering 相关，通常不是算子容差问题，而是地址计算或 lowering 边界错误。

### 修复策略

- 对 load/store 和比较结果加 mask gate，防止 OOB 数据污染后续计算。
- 修正 1D gather、split pointer、wraparound 的边界推导，不在外层维度做多余 modulo。
- 对 InstanceNorm、attention、vdot 等已出现段错误的算子，先缩小到具体指针序列或 layout rewrite，再决定是算子修复还是 lowering 修复。
- 对 index tensor、bitcast、pointer cast 使用正确 bitwidth 和类型分支。
- 当大 shape 触发 LLVM backend crash 时，先检查 block size、reduction 维度和 lowering 产生的 IR 规模。

### 避免误修

- 不要用跳过测试或降低输入规模代替修复越界。
- 不要在上层算子里绕过所有 wraparound；应修正真实边界条件。
- 不要假设 GPU block pointer 语义能直接套到 CPU memref lowering。

### 代表 PR

`!64`/`!95`/`!98` indexer/topk IndexError，`!72` 1D gather crash，`!86` PtrAnalysis boundary extent，`!90` i64 index tensor and bitcast，`!93` split pointer load/store，`!100` empty dynamic offset crash，`!110` wrapping OOB read，`!120` mask gate OOB artifact，`!134` AtomicRMW in unstructured ptr lowering，`!135` attention segfault，`!149` InstanceNorm backward segfault，`!151` large N norm LLVM crash，`!161` vdot segfault，`!163` InstanceNorm 1D split pointer wraparound。

## TritonShared、MLIR 和 LLVM lowering 问题

### 典型现象

- 报 `unexpected op in ptr sequence`、bufferization failed、某个 op 缺少 lowering。
- `tt.atomic_cas`、`AtomicRMWOp`、`tt.extern_elementwise`、Sleef/math op、`MakeTensorPtrOp`、`vector.multi_reduction`、`scf.for` 等在 conversion/pass 中失败。
- 编译通过前端，但在 TritonShared 到 memref、LLVM 或 runtime glue 阶段崩溃。

### 根因判断

按照 IR 流程定位失败边界：Triton IR、TritonShared、StructuredToMemref、TritonToStructured prepass、TPtrToLLVM、Bufferization、LLVM lowering、driver/compiler glue。错误如果明确指向 op 或 pass，就不要先改 FlagGems 算子。

### 修复策略

- 为缺失 op 补齐 lowering 或类型分支，例如 extern/Sleef/math、atomic、pointer cast、bitcast、int64、multi_reduction。
- 修正 structured/unstructured pointer lowering 中对 pointer sequence、dynamic offset、split pointer 的处理。
- 对 layout 非 row-major、transpose、join/insert_slice 等 IR shape 问题，在对应 conversion pass 内保持类型和 rank 一致。
- 对目标特征、LLVM IR optimization 等性能路径，先保持 pipeline 边界清晰，再按 backend 配置启用。

### 避免误修

- 不要在 Python 层吞掉 compiler error；lowering 缺口应在 lowering 层补齐。
- 不要把所有 op 都特判进单个 pass，先确认 op 属于哪个 dialect 和哪个 conversion 阶段。
- 不要让性能 pipeline 改动改变默认语义，除非 PR 明确是性能路径且有指标支撑。

### 代表 PR

`!13`/`!28` bufferization failed，`!15` `tt.expand_dims` axis，`!40`/`!51` extern_elementwise/Sleef，`!43` finite functions，`!47` join operands，`!53` IntToPtr/PtrToInt，`!68` non-row-major transpose，`!90` i64 index and bitcast，`!91` int64 argmax/argmin，`!100` structured `MakeTensorPtrOp`，`!103` atomic_cas float tensor，`!105` atomic pointer types，`!106` i16 prepass crash，`!109` Sleef round mapping，`!121` atomic_cas regression，`!134` AtomicRMW lowering，`!158` backend pipeline，`!159` LLVM IR optimizations。

## Bufferization 与内存生命周期问题

### 典型现象

- bufferization/deallocation pass 失败。
- clone、copy-before-write、dealloc 顺序、alias ownership 或 dangling iterator 导致 crash。
- `scf.for` init/iter_arg 类型不一致，或 IR rewrite 后使用了失效 iterator。

### 根因判断

如果错误出现在 Bufferization、Deallocation、clone lowering、copy-before-write 或 iterator 使用处，优先看 IR 变换顺序和 value ownership。算子层 workaround 往往只能掩盖一条路径，不能修复共享 pass。

### 修复策略

- 调整 clone lowering 与 buffer deallocation 的顺序，保证 dealloc 前后 ownership 清晰。
- 在需要保留原始 buffer 语义时使用 copy-before-write，避免读写别名污染。
- 修正 pass 中 iterator 生命周期，不在 rewrite 后继续使用已失效引用。
- 保持 `scf.for` init value 与 region iter_arg 类型完全一致。

### 避免误修

- 不要用禁用 bufferization/deallocation pass 作为默认修复。
- 不要在单个算子里复制大段临时 buffer 来回避共享 pass bug。
- 不要改掉 IR 类型检查；应让 rewrite 结果满足类型约束。

### 代表 PR

`!13`/`!28` bufferization failed，`!66` dangling iterators，`!71` lower `bufferization.clone` after deallocation，`!92` Bufferization Deallocation，`!102` multinomial copy-before-write，`!104` `scf.for` init/iter_arg type mismatch。

## FlagGems 算子源码问题

### 典型现象

- 某个 FlagGems 测试稳定失败：norm、attention、reduction、topk、multinomial、cumsum、index_add、nonzero、quant、sparse MLA 等。
- 错误文本像算子语义问题：wrong axis、wrong attribute、kernel name syntax error、missing benchmark ATen op。
- 测试或 benchmark 脚本在 CPU backend 下编译失败或参数不适配。

### 根因判断

先判断失败是否只集中在一个算子族。如果是，优先读对应 FlagGems operator、test 和 benchmark 入口；只有当多个算子共享相同 compiler error 时，才升级到 TritonShared 或 runtime 层。

### 修复策略

- 对算子语义错误，修正 axis、attribute、layout、mask、默认值、index 计算或 dtype promotion。
- 对 CPU 不适合 Triton kernel 的局部场景，提供小范围 CPU fallback。
- 对 benchmark/test 编译错误，修正输入生成、测试配置、导入、kernel name、skip list 或 benchmark suite 注册。
- 对随机输入导致的不稳定问题，固定输入或避免随机性，让失败可复现。

### 避免误修

- 不要为了单个算子失败修改全局 JIT 或 lowering，除非能证明根因共享。
- 不要让 benchmark 辅助修复改变 operator 正式语义。
- 不要把随机测试失败当作真实性能或正确性结论。

### 代表 PR

`!4` testcase fixes，`!36` true div，`!38` undefined tensor，`!41` general FlagGems fixes，`!45` builder attribute，`!48` nonzero/cumsum，`!59` constexpr handle，`!61`/`!63`/`!64` indexer，`!75` sparsity test extended to arm，`!77` kernel name，`!78` `torch.bincount`，`!95`/`!98` topk，`!101` reduction，`!108` index_add CPU fallback，`!115` cumsum axis/attribute，`!125` benchmark script，`!139` benchmark ATen ops，`!166`/`!173` random case，`!167` typo，`!170` high priority benchmark suite，`!171` benchmark compile errors。

## 并发、调试和性能问题

### 典型现象

- pytest-xdist 或多进程测试出现文件竞争。
- 缺少 per-kernel debug 信息，难以定位具体 kernel 或 pass。
- 编译时间过长、kernel dispatch overhead 过高、launcher 或 LLVM IR 优化不足。
- OMP、目标特征、llc 配置、launcher `-O3` 等 backend 性能路径需要调整。

### 根因判断

并发问题优先查共享文件、缓存目录、临时文件命名和写入时机。性能问题先确定瓶颈发生在编译阶段、dispatch 阶段、LLVM IR 优化阶段还是运行时执行阶段，再做局部优化。

### 修复策略

- 对多进程文件竞争使用唯一文件、原子写入或更明确的 cache key。
- 增加 per-kernel debug、compile time 和 skip list 能力，让后续失败更容易缩小范围。
- 降低 dispatch overhead 时缓存昂贵检查结果，避免每次 kernel call 重复做全量检查。
- 对编译性能，移除明确拖慢且无必要的 llc option，或启用已验证的 LLVM IR/launcher 优化。
- 对 OMP/目标特征相关优化，保持 backend feature gate，避免影响非目标 CPU。

### 避免误修

- 不要为并发问题引入复杂调度框架；先消除确定的共享资源竞争。
- 不要在没有指标的情况下扩大性能优化范围。
- 不要让 debug 逻辑改变 kernel 编译或执行语义。

### 代表 PR

`!21` gitignore hygiene，`!25` pytest-xdist，`!62` compile stage duplication，`!79` compile time print，`!114` skip list，`!136` per-kernel debugging，`!148` pytest file races，`!150` debug regex，`!147` llc option compile time，`!152` kernel dispatch overhead，`!153` launcher `-O3`，`!157` llc target features，`!159` LLVM IR optimizations，`!170` benchmark suite，`!171` benchmark compile errors，`!172` OMP performance。

## 上游同步、回退和版本兼容问题

### 典型现象

- 上游 Triton/FlagGems 变更引入 CPU backend 不兼容。
- 需要 backport benchmark suite、Triton frontend 语义或基础库。
- 某个历史改动在 CPU backend 下导致 masked load、block pointer 或 test 失败，需要 revert。

### 根因判断

确认当前失败是“上游行为本身变化”还是“CPU backend 未适配新行为”。如果上游变更与 CPU backend 目标冲突，优先保留 CPU 正确性；如果是缺少适配，则补齐对应 backend 路径。

### 修复策略

- Backport 时只带入能解释当前需求的最小改动，并确认 CPU backend 的依赖是否齐全。
- Revert 时说明被回退行为破坏了哪类 CPU 测试或 lowering 语义。
- 对版本语义变化，例如 C `%` semantics，明确选择并更新相关测试或 lowering。
- 基础库集成要检查 runtime headers、Sleef/libdevice、LLVM 版本和 TritonShared 边界。

### 避免误修

- 不要盲目追上游最新行为而牺牲当前 CPU backend 可运行性。
- 不要把 backport 做成大范围同步；越大越难定位回归。
- 不要在没有目标失败的情况下清理历史兼容代码。

### 代表 PR

`!1` LLVM19 CPU adaptation，`!2` runtime headers and Sleef，`!3` ARM matrix/vector support，`!18` decouple original Triton CPU and update Triton，`!19` integrate TritonShared bugfixes，`!22` libdevice/Sleef，`!23` FlagGems v4.2.0，`!50` stale upstream masked load changes，`!57` C semantics for `%`，`!73` non-block pointer flip，`!84` upstream typo fix，`!87` revert non-block pointer flip，`!133` benchmark suite backport。

## AI 修复决策顺序

1. 读失败日志，找首个根因错误：具体测试、op、pass、dtype、shape、backend 和调用栈。
2. 将错误归层：
   - `torch.cuda`、device capability、accelerator access：先查 CPU-only 适配。
   - `Tensor-likes are not close`：先查算子语义、dtype、mask、axis、layout，再查 lowering。
   - segmentation fault、IndexError、OOB、wraparound：先查索引、mask、pointer rewrite 和 layout。
   - bufferization、atomic、extern、Sleef、MakeTensorPtr、scf/vector：先查 TritonShared/MLIR/LLVM lowering。
   - pytest-xdist、benchmark、compile time、dispatch overhead：先查并发、调试或性能路径。
3. 匹配历史 PR 模式：优先从同一错误文本、同一算子族、同一 lowering pass 或同一 backend 配置找代表 PR。
4. 做最小修复：只改能解释当前失败的一层；若需要跨层修改，说明每层承担的职责。
5. 给出验证点：失败用例应通过；若是 compiler 修复，应能看到原失败 op/pass 消失；若是性能修复，应有对应指标；若是随机性修复，应使用确定输入。
6. 检查误修风险：没有放宽容差掩盖错误，没有扩大 backend 假设，没有引入无关重构，没有修改无关测试。

## 快速映射表

| 失败信号 | 优先怀疑层级 | 常用修复方式 | 代表 PR |
| --- | --- | --- | --- |
| `torch.cpu` 缺少 CUDA 风格属性 | Python runtime / FlagGems backend | 替换 CPU 属性或隔离 GPU-only 路径 | `!32`, `!33`, `!34`, `!113` |
| `Tensor-likes are not close` | FlagGems 算子 / dtype / mask / layout | 对齐 reference 语义，修 axis、promotion、mask、rounding | `!58`, `!83`, `!96`, `!123`, `!145` |
| segfault / OOB / wraparound | pointer analysis / indexing / layout | 修 mask gate、split pointer、wraparound、index bitwidth | `!72`, `!110`, `!149`, `!161`, `!163` |
| bufferization/deallocation failed | MLIR bufferization / memory lifecycle | 调整 clone/dealloc/copy-before-write 顺序和 ownership | `!28`, `!71`, `!92`, `!102` |
| atomic / extern / Sleef / MakeTensorPtr lowering error | TritonShared lowering | 补齐 op lowering、类型分支或 pass 处理 | `!40`, `!51`, `!100`, `!103`, `!134` |
| benchmark/test compile error | FlagGems test/benchmark harness | 修输入、注册、导入、skip list、kernel name | `!77`, `!114`, `!125`, `!139`, `!171` |
| 多进程文件竞争 | test infrastructure / cache | 唯一文件、原子写入、明确 cache key | `!25`, `!148` |
| 编译或 dispatch 太慢 | compiler/driver/runtime performance | 缓存昂贵检查、调整 llc/LLVM/launcher/OMP 路径 | `!147`, `!152`, `!153`, `!159`, `!172` |
| 上游同步后回归 | compatibility / backport / revert | 最小 backport 或定向 revert，保留 CPU backend 正确性 | `!50`, `!57`, `!87`, `!133` |
