# Benchmark 系统参考

## 概述

FlagGems benchmark 系统位于 `FlagGems/benchmark/`，用于测量 Triton 算子在 CPU 后端上的性能。采用 pytest 框架驱动，支持多种测量模式、数据类型和形状配置。

## Benchmark 类层级

所有 benchmark 类定义在 `FlagGems/benchmark/performance_utils.py`（789行）。

### 基类：`Benchmark`
- `set_shapes()` — 从 YAML 加载或使用默认形状
- `set_dtypes()` — 按配置过滤数据类型
- `get_latency()` — 根据模式（kernel/operator/wrapper）测量延迟
- `get_input_iter()` — 为每个形状生成输入张量迭代器
- `run()` — 主入口：遍历 dtypes → shapes → 内部测量循环

### 具体子类
| 类名 | 适用算子类型 | 特有指标 |
|------|-------------|---------|
| `GenericBenchmark` | 通用算子 | — |
| `GenericBenchmark2DOnly` | 仅支持2D的算子 | — |
| `BlasBenchmark` | mm, bmm, addmm | TFLOPS |
| `BinaryPointwiseBenchmark` | 二元逐点 | TFLOPS |
| `UnaryPointwiseBenchmark` | 一元逐点 | TFLOPS |
| `UnaryPointwiseOutBenchmark` | 一元逐点（含out参数） | TFLOPS |
| `ScalarBinaryPointwiseBenchmark` | 标量-张量逐点 | TFLOPS |
| `ReductionBenchmark` | 归约算子 | GBPS |
| `NormBenchmark` | 归一化算子 | — |
| `AttentionBenchmark` | 注意力算子 | — |
| `Conv2DBenchmark` | 2D卷积 | — |

## pytest 配置

配置定义在 `FlagGems/benchmark/conftest.py`，通过 `pytest_addoption` 注册：

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `--mode` | `kernel`, `operator`, `wrapper` | 测量模式：kernel=纯Triton内核耗时，operator=完整算子调用，wrapper=Python封装层 |
| `--level` | `comprehensive`, `core` | 测试范围：comprehensive=全部shape，core=核心shape |
| `--dtypes` | 空格分隔的dtype列表 | 如 `float16 float32 bfloat16` |
| `--warmup` | 整数 | 预热迭代次数（默认1000） |
| `--iter` | 整数 | 测量迭代次数（默认100） |
| `--shape_file` | YAML文件路径 | 形状配置文件（默认 `core_shapes.yaml`） |
| `--metrics` | 空格分隔的指标列表 | 要计算的指标 |
| `--record` | `none`, `log` | 是否记录结果到日志文件 |
| `--parallel` | 整数 | 并行度 |
| `--collect-marks` | flag | 收集 pytest marker |

## 形状系统

形状配置文件：`FlagGems/benchmark/core_shapes.yaml`

- 按 `op_name` 或 benchmark 类名索引
- 支持 `shape_desc` 字段描述形状含义
- 模型特定形状：`FlagGems/benchmark/models_shapes/`（如 `qwen25.yaml`）
- 回退默认形状定义在 `attri_util.py::DEFAULT_SHAPES`

## 指标

定义在 `FlagGems/benchmark/attri_util.py::BenchmarkMetrics`（dataclass）：

| 字段 | 含义 |
|------|------|
| `latency_base` | 基线延迟（PyTorch参考实现） |
| `latency` | FlagGems 算子延迟 |
| `speedup` | 加速比 = latency_base / latency |
| `gbps` | 内存带宽（GB/s），归约算子使用 |
| `tflops` | 计算吞吐（TFLOPS），BLAS/逐点算子使用 |
| `accuracy` | 精度对比结果 |
| `error_msg` | 错误信息（如有） |

默认指标：`["latency_base", "latency", "speedup"]`

## CPU 流水线是性能变量

后端默认选择流水线，不需要通过环境变量指定：

- payload 含 matmul 或 batch matmul 时走 SME 流水线。
- 其他 payload 走 SVE 流水线。

这一区分用于解释性能结果，不是 benchmark 的额外配置项。对比结果时必须保持
mode、dtype、shape、warmup、iter、OpenMP 线程数、CPU/NUMA 绑定和缓存状态一致，
再判断流水线内的 lowering、tiling 或代码生成是否造成差异。

SME 路径主要影响矩阵类负载。分析时优先观察 TFLOPS、SVL-aware tiling、
矩阵 lowering、ZA/MOPA 使用和 batch 维度处理。SVE 路径主要影响逐点、归约、
访存和普通向量负载，优先观察 GB/s、向量循环质量、数学函数 lowering 和
尾部处理。

流水线与预期不符时，通过 IR dump 和编译日志确认实际路径，不添加选择流水线的
环境变量。通用调试变量包括：

| 变量 | 作用 |
|------|------|
| `TRITON_SHARED_DUMP_PATH=<path>` | dump 各阶段 MLIR 到指定目录 |
| `MLIR_ENABLE_DUMP=1` | 启用各阶段 IR 打印 |
| `TRITON_PRINT_COMPILE_TIME=1` | 打印各阶段编译耗时 |

实现入口位于 `triton-shared/backend/compiler.py` 的 `CPUBackend`。性能专项的
执行方式和历史证据分别由
[`sme-benchmark`](../../skills/sme-benchmark/SKILL.md) 和
[`silu-pointwise`](../../skills/silu-pointwise/SKILL.md) skill 保存。

## 执行模式

### 单算子 benchmark
```bash
pytest FlagGems/benchmark/test_add.py -x -v --mode kernel --level core --dtypes float16 float32
```

### 优先级套件
```bash
python FlagGems/benchmark/run_priority_suite.py --mode kernel --jobs 4
```
执行 `PRIORITY_OPS` 中定义的 70+ 核心算子，支持多进程并行。

### 模型 benchmark
```bash
# 离线吞吐测试
bash FlagGems/benchmark/models_benchmark/offline.sh

# 在线服务测试（需 FlagScale/vLLM）
bash FlagGems/benchmark/models_benchmark/online_with_gems.sh
```

## 结果日志

结果输出到 `FlagGems/benchmark/results/`，格式为每行一个 JSON：
```
[INFO] {"op_name": "add", "dtype": "float32", "shape": "(1024, 1024)", "latency": 0.123, "speedup": 1.5}
```

使用 `FlagGems/benchmark/summary_for_plot.py` 进行结果聚合和对比。

## Benchmark 测试文件命名

每个 `test_*.py` 对应一个或一组算子，使用 pytest marker 标记：
```python
@pytest.mark.add
def test_add_benchmark():
    bench = BinaryPointwiseBenchmark("add", torch.add, dtypes=FLOAT_DTYPES)
    bench.run()
```

Marker 用于 `benchmark_for_models.py` 按模型 shape 选择算子执行。
