# FlagGems Benchmark TFLOPS 统计口径

## 目录

- [背景](#背景)
- [当前实现原则](#当前实现原则)
- [FlagGems 当前原生 TFLOPS 方式](#flaggems-当前原生-tflops-方式)
- [已有原生公式](#已有原生公式)
- [profiling 补齐策略](#profiling-补齐策略)
- [推荐补齐公式](#推荐补齐公式)
- [示例](#示例silu_and_mul)
- [维护约定](#维护约定)

## 背景

`TFLOPS` 用来表示算子的理论计算吞吐量：

```text
TFLOPS = FLOPs / elapsed_seconds / 1e12
       = FLOPs / latency_ms * 1e3 / 1e12
```

这里的 `FLOPs` 通常不是硬件真实执行的浮点指令数，而是按算子数学定义或 benchmark 约定得到的理论计算量。因此它适合在同一口径下做参考吞吐对比，不适合解释实现内部到底执行了多少条指令。

对于 GEMM、BMM、Conv 这类算子，业界公式较稳定。对于 pointwise、activation、softmax、normalization、fused op 等算子，FLOPs 口径并不完全统一，需要固定公式并保留来源信息。

## 当前实现原则

profiling 脚本最终只输出一个统一指标：

```text
tflops
```

不额外暴露 `derived_tflops` 列。区别通过元信息追溯：

```json
{
  "tflops": 0.0000202,
  "flops": 16384,
  "tflops_source": "derived",
  "flops_formula_id": "derived:fused_pointwise"
}
```

`tflops_source` 的含义：

| 值 | 含义 |
| --- | --- |
| `benchmark` | 来自 FlagGems benchmark 原生 `record.log` 中的 `tflops` |
| `derived` | profiling 聚合阶段按静态公式补齐 |
| `none` | 原生值为空，且当前没有可用公式 |

补齐逻辑发生在结果聚合阶段，也就是 pytest benchmark 已经结束、`record.log` 已经生成之后。该阶段只读取 `op_name`、`shape_detail`、`latency`、`tflops` 等已有字段，并做纯 Python 整数乘除计算，不调用算子、不分配大 tensor，不影响性能计时。

## FlagGems 当前原生 TFLOPS 方式

FlagGems benchmark 的原生计算位于：

```text
FlagGems/benchmark/performance_utils.py
```

当前计算形式为：

```text
metric.tflops = get_tflops(...) / metric.latency / 1e12 * 1e3
```

也就是：

```text
benchmark 类定义的理论 FLOPs / Gems latency
```

原生 `DEFAULT_METRICS` 只包含：

```text
latency_base, latency, speedup
```

只有 benchmark 类把 `tflops` 加入 `DEFAULT_METRICS`，或者命令行显式启用 `--metrics tflops`，才会尝试产出原生 `tflops`。

## 已有原生公式

以下公式来自现有 FlagGems benchmark 类，profiling 侧会将其登记为 `benchmark` 来源。

| 类型 | 公式 | 说明 |
| --- | ---: | --- |
| `mm` | `2 * M * N * K` | GEMM 标准口径 |
| `bmm` | `2 * B * M * N * K` | Batched GEMM |
| `addmm` | `M * N * (2 * K + 1)` | Matmul 加 bias/add |
| `baddbmm` | `B * M * N * (2 * K + 1)` | Batched matmul 加 bias/add |
| `grouped_mm` | `sum_g(2 * M_g * N_g * K_g)` | grouped matmul |
| unary pointwise | `numel` | 现有 `UnaryPointwiseBenchmark` |
| binary pointwise | `2 * numel` | 现有 `BinaryPointwiseBenchmark` |
| scalar binary pointwise | `numel` | 现有 `ScalarBinaryPointwiseBenchmark` |
| Tex GLU forward | `input_numel` | `geglu/glu/reglu/swiglu` |
| Tex GLU backward | `2 * input_numel` | `dgeglu/dglu/dreglu/dswiglu` |

注意：部分算子继承了带 `tflops` 的 benchmark 类，但没有覆盖对应 `op_name` 的公式。例如某些 BLAS-like 算子如果没有自己的 `get_tflops`，原生输出可能为 0 或 `null`。这类情况由 profiling 聚合阶段补齐。

## profiling 补齐策略

当原生 `record.log` 中 `tflops` 为 `null` 时，profiling 聚合阶段会查 `scripts/profiling/flops.py` 中的公式目录补齐：

```text
flops = formula(op_name, shape_detail)
tflops = flops / latency_ms * 1e3 / 1e12
```

默认 `summary.md` 仍只显示 `TFLOPS`。公式详情不默认进入 markdown 表格，可以通过独立命令查看：

```bash
PYTHONPATH=$AGENT_DIR/AgentForTritonCPU/skills/flaggems-benchmark-profile/scripts \
python3 -m profiling.flops

PYTHONPATH=$AGENT_DIR/AgentForTritonCPU/skills/flaggems-benchmark-profile/scripts \
python3 -m profiling.flops --json
```

## 推荐补齐公式

### BLAS-like

| 算子 | 推荐 FLOPs |
| --- | ---: |
| `mv` | `2 * M * N` |
| `addmv` | `M * (2 * N + 1)` |
| `outer` | `M * N` |
| `addr` | `2 * M * N` |
| `dot` / `vdot` | `2 * N` |
| `kron` | `numel(input0) * numel(input1)` |

复杂数 `vdot` 的真实数学 FLOPs 可以更高，但当前补齐公式采用 `2 * N` 作为统一参考口径。若后续需要区分 complex dtype，可在公式目录中单独扩展。

### Convolution

| 算子 | 推荐 FLOPs |
| --- | ---: |
| `conv1d` | `2 * N * Cout * Lout * (Cin / groups) * K` |
| `conv2d` | `2 * N * Cout * Hout * Wout * (Cin / groups) * Kh * Kw` |
| `conv3d` | `2 * N * Cout * Dout * Hout * Wout * (Cin / groups) * Kd * Kh * Kw` |
| `conv_depthwise2d` | `2 * N * C * Hout * Wout * Kh * Kw` |

### Pooling

| 算子 | 推荐 FLOPs |
| --- | ---: |
| `avg_pool2d` | `output_numel * kernel_h * kernel_w` |
| `max_pool2d_with_indices` | `output_numel * kernel_h * kernel_w` |

这里的 pooling FLOPs 是参考工作量，不区分加法、比较和除法的精确代价。

### Pointwise / Activation / Fused

| 类型或算子 | 推荐 FLOPs |
| --- | ---: |
| unary pointwise | `numel` |
| binary pointwise | `2 * numel`，沿用 FlagGems 当前口径 |
| scalar binary pointwise | `numel` |
| `addcmul` / `addcdiv` | `3 * numel` |
| `silu_and_mul` | `4 * numel` |
| `gelu_and_mul` | `6 * numel` |
| `where` | `numel` |

对于 `exp/log/sin/tanh/gelu/silu` 等特殊函数，不建议尝试展开底层近似实现，否则会让结果变成实现相关成本。这里统一采用固定每元素系数。

### Reduction / Softmax

| 算子 | 推荐 FLOPs |
| --- | ---: |
| `sum` / `prod` | `input_numel` |
| `max` / `min` / `amax` / `amin` | `input_numel` |
| `all` / `any` / `argmax` / `argmin` | `input_numel` |
| `mean` | `2 * input_numel` |
| `var` | `3 * input_numel` |
| `std` | `4 * input_numel` |
| `softmax` / `log_softmax` / `safe_softmax` | `4 * input_numel` |
| `scaled_softmax` | `5 * input_numel` |

这类公式用于参考吞吐。比较、指数、除法等操作没有统一的硬件等价 FLOPs 口径。

### Normalization

| 算子 | 推荐 FLOPs |
| --- | ---: |
| `rms_norm` | `4 * input_numel` |
| `layer_norm` | `5 * input_numel` |
| `group_norm` | `5 * input_numel` |
| `instance_norm` | `5 * input_numel` |
| `batch_norm` | `5 * input_numel` |
| `batch_norm_backward` | `8 * input_numel` |
| `skip_layer_norm` | `6 * input_numel` |
| `fused_add_rms_norm` | `6 * input_numel` |
| `vector_norm` | `2 * input_numel` |

### Attention

对 attention 默认采用 QK 和 PV 两个主 matmul 项：

```text
QK^T: 2 * B * H * M * N * D
PV:   2 * B * H * M * N * D

total = 4 * B * H * M * N * D
```

当前覆盖：

```text
scaled_dot_product_attention
flash_attention_forward
flash_mla
```

softmax、scale、mask 等开销没有计入默认 attention FLOPs。这样更接近业界常用的 attention 主计算量口径。

## 示例：silu_and_mul

`silu_and_mul` 当前使用 `GenericBenchmark`，benchmark 原生 `record.log` 中 `tflops` 为 `null`。

聚合阶段使用补齐公式：

```text
silu_and_mul(x, y) = silu(x) * y
FLOPs = 4 * numel
```

例如 shape 为 `[64, 64]` 时：

```text
numel = 4096
flops = 4 * 4096 = 16384
tflops = 16384 / latency_ms * 1e3 / 1e12
```

对应 JSON 元信息类似：

```json
{
  "tflops": 0.00002023164757390978,
  "flops": 16384,
  "tflops_source": "derived",
  "flops_formula_id": "derived:fused_pointwise"
}
```

## 维护约定

1. 默认 `summary.md` 只显示统一 `TFLOPS`，不展开公式细节。
2. `summary.json` 保留 `flops`、`tflops_source`、`flops_formula_id` 以便追溯。
3. 原生 `tflops` 优先，不覆盖 benchmark 已经给出的值。
4. 原生值为空时才按静态公式补齐。
5. 公式只允许依赖 `record.log` 已有字段，不进入 benchmark 计时路径。
6. 新增公式时同步更新 `scripts/profiling/flops.py` 和本文档。
