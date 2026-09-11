# 项目地图

## 仓库概况

工作区由 `AgentForTritonCPU` tooling/spec 仓和 `triton-cpu` 产品仓组成。
Triton CPU 基于 Triton 3.2.0，通过 **TritonShared + LLVM** 将 Triton
语言编译到 CPU 上执行。主要目标是 Kunpeng（AArch64/ARM）。

## 顶层目录

```text
$AGENT_DIR/
├── AGENTS.md -> AgentForTritonCPU/agent/AGENTS.md
├── AgentForTritonCPU/             # Agent rules/specs/scripts
│   ├── agent/
│   └── skills/
└── triton-cpu/                    # 产品仓根目录
├── FlagGems/                      # ~360 个 Triton 算子实现 + benchmark
│   ├── src/flag_gems/
│   │   ├── __init__.py            # 算子注册，enable()/use_gems() 入口
│   │   ├── ops/                   # 单算子实现（add.py, mm.py, attention.py...）
│   │   ├── fused/                 # 融合算子
│   │   ├── modules/               # 模块级封装（RMSNorm, RoPE...）
│   │   ├── utils/
│   │   │   └── libentry.py        # LibTuner / LibEntry 自调优基础设施
│   │   ├── runtime/
│   │   │   ├── __init__.py        # device 检测，backend 绑定
│   │   │   └── backend/
│   │   │       ├── _arm/          # ARM CPU 后端（tune_configs, heuristics, ops）
│   │   │       ├── _kunpeng/      # Kunpeng 特定覆盖
│   │   │       └── ...            # 其他 vendor 后端
│   │   └── testing/               # 测试工具
│   ├── benchmark/
│   │   ├── performance_utils.py   # Benchmark 类层级（789行）
│   │   ├── conftest.py            # pytest 配置（--mode, --level, --dtypes...）
│   │   ├── attri_util.py          # 指标、模式、级别枚举
│   │   ├── run_priority_suite.py  # 70+ 优先级算子执行器
│   │   ├── core_shapes.yaml       # 形状配置（按算子索引）
│   │   ├── test_*.py              # 250+ 单算子 benchmark 测试
│   │   └── models_benchmark/      # 模型级 benchmark
│   └── tests/                     # 正确性测试
├── triton-shared/                 # Triton Shared 中间层（CPU 后端关键路径）
│   ├── backend/
│   │   └── compiler.py            # CPUBackend 主编译器与流水线调度
│   ├── lib/
│   │   ├── Conversion/            # MLIR conversion passes
│   │   │   ├── TritonToLinalg/    # Triton IR → Linalg
│   │   │   ├── TritonToStructured/
│   │   │   ├── StructuredToMemref/
│   │   │   ├── UnstructuredToMemref/
│   │   │   └── TPtrToLLVM/
│   │   ├── Analysis/
│   │   │   ├── PtrAnalysis.cpp    # 指针分析（55K）
│   │   │   └── MaskAnalysis.cpp
│   │   ├── AnalysisStructured/
│   │   │   └── PtrAnalysis.cpp    # 新结构化指针分析（80K）
│   │   └── Dialect/               # TritonStructured, TPtr 等自定义 dialect
│   ├── tools/
│   │   └── triton-shared-opt/     # 独立 MLIR 优化工具
│   └── test/                      # Lit 测试
├── lib/                            # C++ Triton dialect 和 conversion（上游）
├── include/                        # C++ 头文件
├── python/                         # Python 前端（triton 包）
│   ├── triton/
│   │   ├── compiler/              # compiler.py, code_generator.py
│   │   ├── language/              # core.py（96K）, semantic.py（81K）
│   │   ├── runtime/               # jit.py, driver.py, autotuner.py
│   │   └── backends/              # 后端发现系统
│   ├── src/                       # C++ 绑定（ir.cc 76K）
│   └── test/                      # Python 测试套件
├── test/                           # Lit 测试（C++/MLIR）
├── unittest/                       # GoogleTest C++ 单测
├── cmake/                          # CMake 模块 + LLVM hash
├── CMakeLists.txt                  # 顶层构建（C++17, LLVM/MLIR 依赖）
└── .github/workflows/              # CI 工作流
```

## 关键环境变量

| 变量 | 作用 | 使用场景 |
|------|------|---------|
| `TRITON_USE_SHARED_BACKEND=1` | 激活 TritonShared CPU 后端 | 运行共享 CPU 后端时设置；环境脚本会默认导出 |
| `TRITON_SHARED_DUMP_PATH=<path>` | 输出各阶段 MLIR dump | 调试 lowering 问题 |
| `TRITON_PRINT_COMPILE_TIME=1` | 打印各阶段编译耗时 | 性能分析 |
| `LLVM_BINARY_DIR=<path>` | LLVM 工具链路径 | 编译时需要 llc 等工具 |
| `TRITON_SHARED_OPT_PATH=<path>` | triton-shared-opt 二进制路径 | 指定优化工具位置 |
| `OMP_NUM_THREADS=<N>` | OpenMP 线程数 | 控制 CPU 并行度 |
| `GOMP_CPU_AFFINITY` / `KMP_AFFINITY` | CPU 线程绑定 | 可复现性能测试 |
| `MLIR_ENABLE_DUMP=1` | 启用 MLIR pass dump | 调试 pass pipeline |

## 关键类

### FlagGems Benchmark（`FlagGems/benchmark/performance_utils.py`）
- `Benchmark` — 基类：管理 dtypes、shapes、延迟测量、结果输出
- `GenericBenchmark` — 通用算子 benchmark
- `BlasBenchmark` — BLAS 算子（mm, bmm），添加 TFLOPS 指标
- `BinaryPointwiseBenchmark` / `UnaryPointwiseBenchmark` — 逐点算子
- `ReductionBenchmark` — 归约算子，添加 GBPS 指标
- `NormBenchmark` — 归一化算子

### TritonShared 编译器（`triton-shared/backend/compiler.py`）
- `CPUOptions` — 编译器选项（frozen dataclass）：debug, arch, num_warps, num_threads, num_ctas, num_stages
- `CPUBackend` — 主编译器类，继承 `triton.backends.compiler.BaseBackend`
  - `_optimize_ttsharedir()` — 流水线调度

### FlagGems Runtime（`FlagGems/src/flag_gems/runtime/`）
- `DeviceDetector` — 设备检测单例
- `ConfigLoader` — 后端配置加载
- `LibTuner` / `LibEntry` — 自调优基础设施（`utils/libentry.py`）

## 构建系统

- **CMake**: C++17，依赖 LLVM/MLIR，支持插件后端
- **Python setuptools**: `python/setup.py` 构建 triton 共享库
- **LLVM**: 固定在 `cmake/llvm-hash.txt` 中的 commit
- **插件**: `TRITON_PLUGIN_DIRS` 环境变量加载 triton-shared
