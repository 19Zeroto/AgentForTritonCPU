# FlagGems fusion metrics 目标设计

本文定义 `flaggems-fusion-metrics` skill 的最终目标架构。当前工作区已按该架构
迁移到外部公式、固定映射和日志后处理；以下内容同时作为后续公式维护和回退审计
的约束。

## 目录

- [目标与边界](#目标与边界)
- [总体架构](#总体架构)
- [Skill 文件布局](#skill-文件布局)
- [串行 benchmark runner](#串行-benchmark-runner)
- [固定输入映射表](#固定输入映射表)
- [外部公式计算](#外部公式计算)
- [日志后处理与报告](#日志后处理与报告)
- [结果更新与回退](#结果更新与回退)
- [产品仓迁移](#产品仓迁移)
- [错误处理](#错误处理)
- [实施顺序](#实施顺序)
- [验收标准](#验收标准)

## 目标与边界

目标：

- FlagGems benchmark 只测量并记录 Torch/Gems 时延、dtype 和完整
  `shape_detail`。
- FLOPs 和 logical bytes 完全离开 benchmark 测量进程。
- 固定 benchmark 输入预先计算为映射表；报告导出只查表。
- 支持串行批量执行多个算子并集中保存日志。
- CPU NUMA 节点与 memory/HBM NUMA 节点可独立绑定。
- 新日志可生成完整报告，也可按算子替换已有报告。
- 保持现有 compute/memory CSV 列顺序和计算口径，明确批准的口径变化除外。

非目标：

- 不统计硬件实际执行指令、cache line 或 DDR/HBM 真实流量。
- 不在 runner 中进行公式计算、报告生成、profiling 或性能诊断。
- runner 不选择或强制 SME/SVE pipeline；报告默认按类别使用 compute=SME、
  memory=SVE，也可用显式参数覆盖报告目标。
- 不自动 fallback dtype，不重试失败 benchmark。
- 不并发运行性能测试；任意时刻最多存在一个 benchmark 子进程。
- 不通过随机输入的实际内容构造日志查询 key。

本文中的 FLOPs 和 bytes 均为定义好的 logical metric，不代表硬件计数器结果。

## 总体架构

```text
benchmark_cases.yaml
        │
        v
串行 benchmark runner
        │
        ├── records/*.log
        ├── stdout/*.log
        └── run.json
                 │
                 v
            日志后处理器 <── fusion_metrics_map.json
                 │
                 ├── fused_compute_tflops.csv
                 ├── fused_non_compute_bandwidth.csv
                 ├── coverage.csv
                 └── run.json

固定 benchmark 日志/输入
        │
        v
外部公式计算器 <── fusion_metrics_formulas.yaml
        │
        v
fusion_metrics_map.json
```

三个阶段职责隔离；runner 在 benchmark 子进程结束后以独立进程调用报告脚本：

1. runner 执行 benchmark 并收集原始日志，随后调用 `fusion_metrics.py report`。
2. 映射维护阶段单独计算固定输入的 logical FLOPs/bytes。
3. 报告阶段只解析日志、查询映射表和计算展示列。

## Skill 文件布局

目标布局：

```text
skills/flaggems-fusion-metrics/
├── SKILL.md
├── agents/
│   └── openai.yaml
├── references/
│   └── design.md
└── scripts/
    ├── run_fusion_benchmarks.py
    ├── fusion_metrics.py
    ├── benchmark_cases.yaml
    ├── fusion_metrics_formulas.yaml
    └── fusion_metrics_map.json
```

职责：

- `run_fusion_benchmarks.py`：串行执行 pytest benchmark 命令，并在结束后调用
  `fusion_metrics.py report` 生成本次报告。
- `fusion_metrics.py build-map`：单独计算或更新固定输入映射表。
- `fusion_metrics.py report`：从一个或多个日志生成完整报告。
- `fusion_metrics.py update-report`：按算子更新已有报告。
- `benchmark_cases.yaml`：测试文件、marker、预定 dtype 等命令元数据。
- `fusion_metrics_formulas.yaml`：公式、输入类型和特殊固定 workload 规则。
- `fusion_metrics_map.json`：固定输入到 logical FLOPs/bytes 的生成结果。

`SKILL.md` 仍是统一 AI 入口。详细架构只保存在本文，不在 `SKILL.md` 重复。

## 串行 benchmark runner

### 职责

runner 在 benchmark 阶段只执行以下工作，结束后再调用独立的报告进程：

1. 读取算子选择和固定命令元数据。
2. 拼接标准 pytest benchmark 命令。
3. 校验 CPU NUMA、memory NUMA、CPU list 和 OMP 线程数。
4. 按算子顺序启动子进程；前一个结束后才启动下一个。
5. 保存 stdout、FlagGems record log、命令和退出状态。
6. 输出本次集中日志目录。
7. 调用 `fusion_metrics.py report --logs <run-dir>`，将结果写入
   `<run-dir>/report`。

runner 不在 benchmark 子进程中加载 formula YAML 或映射表，不计算展示指标；
报告失败会记录到 runner 的 `run.json`，并使本次 runner 返回非零状态。

### 单 job 约束

- 不提供 `--jobs`，或仅接受固定值 `1`。
- 任意时刻最多运行一个 pytest benchmark 子进程。
- 多算子按确定顺序串行执行。
- 一个算子失败后记录状态，继续后续算子；最终退出码反映是否存在失败。
- 不自动重跑，不更换 dtype，不修改 benchmark 参数。

串行执行避免 CPU、LLC、内存带宽和 HBM 带宽在多个性能测试之间竞争。该约束比
批量吞吐优先级更高。

### CPU 与 memory/HBM 绑定

CPU 和内存节点独立指定：

```text
--cpu-node <CPU NUMA node list>
--mem-node <DDR/HBM NUMA node list>
--omp-threads <N>
--cpu-list <optional explicit CPUs>
```

命令前缀：

```bash
numactl \
  --cpunodebind=<cpu-node> \
  --membind=<mem-node> \
  --physcpubind=<cpu-list>
```

`--cpu-list` uses numactl's range syntax such as `300-331` or
`300-331,400-403`. `--mem-node` accepts comma-separated nodes and ranges such
as `20,22,23,27`; numactl supports binding multiple memory nodes in one
`--membind` option. The runner validates that every selected node exists and
has memory before starting the benchmark.

约束：

- CPU node 必须存在可用 CPU。
- memory node 必须存在可用内存；允许它是没有 CPU 的 HBM-only node。
- 未显式给出 CPU list 时，从 CPU node 的 `cpulist` 选择前
  `OMP_NUM_THREADS` 个 CPU。
- `OMP_NUM_THREADS` 不得超过绑定 CPU 数量，且最大为 32。
- `--cpunodebind` 与 `--membind` 不要求相同节点。
- runner 不做内存预热、page migration、带宽探测或其他隐式操作。

### Benchmark 命令

每个算子依赖 benchmark 自身的默认指标集合（当前为
`latency_base`、`latency`、`speedup`），runner 不再重复传递 `--metrics`。
默认 shape file 也由 benchmark 自身选择；只有用户显式传入自定义 shape file
时才追加 `--shape_file`。

命令结构：

```bash
python3 -m pytest \
  benchmark/<test-file> \
  -q --tb=no \
  -m <marker> \
  --mode <mode> \
  --level <level> \
  --warmup <warmup> \
  --iter <iter> \
  --dtypes <configured-dtype> \
  --record log
```

禁止 runner 添加：

```text
--metrics tflops
--metrics gbps
FLAGGEMS_FUSION_METRICS_CONFIG
FLAGGEMS_FUSION_METRICS_KEY
TRITON_SHARED_FORCE_SME_PIPELINE
TRITON_SHARED_FORCE_SVE_PIPELINE
FLAGGEMS_CACHE_DIR
TRITON_CACHE_DIR
```

runner 继承调用者已准备好的环境，只显式设置 `OMP_NUM_THREADS`。若父环境已经
存在 fusion metric 注入变量或强制 pipeline 变量，preflight 直接失败，不静默
清除后继续执行。

### dtype 与算子命令元数据

`benchmark_cases.yaml` 保存每个算子的确定命令信息：

```yaml
silu_and_mul:
  test_file: test_silu_and_mul.py
  marker: silu_and_mul
  dtype: float32
flash_mla:
  test_file: test_flash_mla.py
  marker: flash_mla
  dtype: bfloat16
```

当前默认值按类别设置：compute 算子使用 bfloat16，memory 算子使用
float32；用户仍可通过 `--dtypes` 显式覆盖。

用户显式 dtype 覆盖与固定默认值均在启动前确定。运行失败后不得切换 BF16 或取消
dtype 重新执行。不同 dtype 的结果必须来自明确的独立运行。

runner 在 Python 文件内维护当前默认的 compute 与 memory 两个算子列表，并提供
`--suite compute|memory|all` 选择；默认 `all` 按合并列表顺序串行执行。传入
`--ops` 时仅执行显式选择的子集，不能与 `--suite` 同时使用。

当前公式配置中启用的算子为 25 个：compute 6 个、memory 19 个；3 个 disabled
公式不加入 runner 默认列表。

### 日志目录

FlagGems record logger 使用进程当前目录创建相对日志。runner 为每个算子设置独立
的仓外工作目录，并传入短的相对 test/shape 路径，无需修改产品 logger。执行工作目录
使用短临时路径，避免 pytest 根据命令参数拼出的 record 文件名超过文件系统长度
限制；结果日志仍归档到集中输出目录。

```text
$AGENT_DIR/logs/fusion_metrics/benchmark_runs/<run-id>/
├── run.json
├── records/
│   ├── silu_and_mul.log
│   └── flash_mla.log
├── stdout/
│   ├── silu_and_mul.log
│   └── flash_mla.log
├── report/
│   ├── fused_compute_tflops.csv
│   ├── fused_non_compute_bandwidth.csv
│   ├── coverage.csv
│   └── run.json
└── work/
    ├── silu_and_mul/
    └── flash_mla/
```

`run.json` 仅记录执行事实：

- 生成时间。
- mode、level、warmup、iter、shape file、OMP 线程数。
- CPU node、memory node、CPU list。
- 每个算子的 dtype、完整命令、退出码、状态和日志相对路径。
- 报告调用状态、退出码和报告输出日志相对路径。

runner manifest 不记录公式、target、映射表或 pipeline 推断；报告自身的
`run.json` 记录报告生成事实。

## 固定输入映射表

### Key

映射 key 使用日志能够完整重建的 benchmark 调用描述：

```python
key = canonical_json([
    canonical_op_name,
    canonical_dtype,
    complete_shape_detail,
])
```

规范化规则：

- alias 转主 `op_name`。
- `torch.float32`、`fp32` 等统一为 `float32`。
- tuple 转 JSON list。
- kwargs key 排序。
- 保留 bool、数字、字符串和 null。
- JSON 使用紧凑分隔符，不添加空格。
- 不使用 hash，不把 Tensor 实际值放入 key。

完整 `shape_detail` 已包含位置参数和 kwargs；同一 shape 下的 causal、window、
reduction、cache dtype 等调用差异因此保持不同 key。dtype 必须属于 key，因为相同
shape 的 logical bytes 随 dtype 改变。

### 文件格式

```json
{
  "generated_at": "2026-08-07T10:00:00+08:00",
  "entries": {
    "[\"silu_and_mul\",\"float32\",[[4096,4096],[4096,4096]]]": {
      "category": "memory",
      "metric": "logical_bytes",
      "value": 201326592
    }
  }
}
```

映射表只保留表级 `generated_at`，不添加 SHA、schema version 或公式版本字段。

维护规则：

- 新输入：允许增量添加。
- 相同 key：新计算值覆盖旧值。
- 公式或 benchmark 输入构造变化：人工执行全量重建。
- 缺少 key：报告严格失败，不以 `N/A` 发布部分数据。
- 写入使用临时文件和原子替换。

映射表是公式计算器生成的物化结果，不手工编辑数值。

## 外部公式计算

### 可行性

公式计算器可在不修改、也不导入 `triton-cpu` 运行时注入代码的前提下计算固定
logical metric。它从日志取得完整 `shape_detail` 和 benchmark dtype，使用元数据
对象计算：

```text
TensorMeta = shape + element_size
```

计算器不创建真实大 Tensor，不运行 Torch/FlagGems 算子，不进入性能测量进程。

memory 公式通过输入类型规则确定每个 Tensor 的元素字节数：

- benchmark dtype。
- 固定 bool/int32/int64。
- 与另一个输入相同。
- optional/None。

### 特殊固定 workload

`flash_attn_varlen_func`：

- 当前 benchmark 的 `cu_seqlens_q` 和 `seqused_k` 是固定列表。
- formula 配置保存这些固定 workload 参数。
- 完整调用 key 选择对应列表并精确计算 logical FLOPs。
- 未知 key 直接报告缺少固定 workload 定义。

`flash_mla`：

- formula 配置保存 `cache_seqlens[b] = L0 + 2b` 等固定生成规则。
- 从固定调用描述取得 B、Sq、Hq、D、Dv 和 L0 后精确求和。
- 不创建 query/cache Tensor。

`rwkv_mm_sparsity`：

- 当前实际 `count_nonzero(k)` 随随机输入变化，不能作为固定 key 的稳定结果。
- 映射表改用当前生成逻辑对应的固定理论非零量：

```text
target_sparsity = 0.9
expected_nnz = numel(k) * (1 - target_sparsity) / 2
logical_flops = 2 * expected_nnz * value.shape[1]
```

- 该值是固定理论 FLOPs，不声称等于每次随机 Tensor 的实际 nnz。

除 `rwkv_mm_sparsity` 的明确口径变化外，迁移阶段机械保留现有公式和 pass-count
约定，不同时修正其他公式审计项。公式正确性修订使用独立任务。

### 映射命令

全量重建：

```bash
python3 fusion_metrics.py build-map \
  --logs <fixed-log-or-run-dir> \
  --output-map <fusion_metrics_map.json>
```

增量更新：

```bash
python3 fusion_metrics.py build-map \
  --logs <new-log-or-run-dir> \
  --ops silu_and_mul flash_mla \
  --update-map <fusion_metrics_map.json>
```

任一 workload 计算失败时不发布新映射表。

## 日志后处理与报告

### 输入

报告器接受：

- runner 输出的完整 `<run-id>/` 目录；或
- 用户显式指定的一个或多个 FlagGems record log。

优先使用 run 目录，因为 `run.json` 提供日志本身缺失的 warmup、iter、OMP 和
NUMA/membind 信息。直接读取零散日志时，未知元数据写 `N/A`，不得伪造默认值。

### 处理流程

```text
读取 record JSON
  -> 提取 op_name/dtype/mode/level/shape_detail/latency
  -> 构造规范化 key
  -> 查询 fusion_metrics_map.json
  -> 计算展示指标
  -> 生成 compute/memory CSV
```

报告阶段不执行公式，不 import Torch/FlagGems，不调用 benchmark。

### 派生列

Compute：

```text
measured_tflops = logical_flops / latency_ms / 1e9
expected_latency_ms = logical_flops / target_tflops / 1e9
achievement_percent = expected_latency_ms / latency_ms * 100
torch_speedup = latency_base_ms / latency_ms
```

Memory：

```text
measured_gbps = logical_bytes / latency_ms / 1e6
torch_gbps = logical_bytes / latency_base_ms / 1e6
expected_latency_ms = logical_bytes / target_gbps / 1e6
achievement_percent = expected_latency_ms / latency_ms * 100
speedup = latency_base_ms / latency_ms
```

默认目标保持：

- SME/float32：2.52631579 TFLOPS。
- SME/bfloat16：5.05263158 TFLOPS。
- SVE/float32：0.71578947 TFLOPS。
- memory：30.72 GB/s。

pipeline 字段是报表目标配置，不由 runner 强制，也不能据此声称实际编译流水线。
报告未指定 `--pipeline` 时按算子类别选择 compute=SME、memory=SVE；指定后则
对所有类别使用该 pipeline。若运行记录未明确提供实际 pipeline，报告中的该列
仍表示采用的目标 profile，而不是实际编译流水线。

### 输出

继续保留：

```text
fused_compute_tflops.csv
fused_non_compute_bandwidth.csv
coverage.csv
run.json
runs/<run-id>/
```

`coverage.csv` 状态：

- `READY`
- `DISABLED`
- `MAP_MISSING`
- `LOG_MISSING`
- `INVALID_LOG`
- `FORMULA_ERROR`

报告可记录映射表的 `generated_at`，但不写公式、映射表或输入日志的 SHA/版本号。

## 结果更新与回退

### 完整报告

```bash
python3 fusion_metrics.py report \
  --logs <benchmark-run-dir>
```

`--map` 默认读取脚本同目录的 `fusion_metrics_map.json`，`--output-dir` 默认
写入 `<benchmark-run-dir>/report`；需要时可显式覆盖。未指定 `--pipeline` 时，
compute 行使用 SME 目标，memory 行使用 SVE 目标。

所有日志 key 命中映射表后，先在新的 `runs/<run-id>/` 快照生成完整文件，再原子
发布顶层结果。

### 按算子替换

```bash
python3 fusion_metrics.py update-report \
  --logs <new-benchmark-run-dir> \
  --ops silu_and_mul flash_mla \
  --output-dir <existing-report-dir>
```

更新规则：

1. 新日志必须包含全部所选算子。
2. 所选算子的每一行都必须命中映射表。
3. 先完整生成新行，不边解析边修改旧文件。
4. 删除旧 compute/memory CSV 中所选算子的全部行。
5. 插入新行；未选算子保持不变。
6. 创建新快照后再原子发布顶层文件。
7. 任一步失败时保留旧顶层结果不变。

同一批新日志内出现重复 workload key 时默认报错，不使用文件顺序静默选择一条。

旧 `runs/<run-id>/` 快照不删除。回退通过重新发布选定旧快照完成，不修改 Git 历史。

## 产品仓迁移

当前注入来自产品仓 commit `df2dc2603`。目标迁移需处理四个文件：

- 恢复 `FlagGems/benchmark/performance_utils.py` 的原始指标路径。
- 恢复 `FlagGems/benchmark/attri_util.py`，移除注入专用 logical 字段。
- 删除 `FlagGems/benchmark/fusion_metrics.py`。
- 将公式迁入 skill 后删除
  `FlagGems/benchmark/fusion_metrics_formulas.yaml`。

保留此前融合算子 benchmark 支持 commit `23e52ff` 和 `69c1ac6`。不回退测试文件、
算子实现或已有原始日志。

产品仓回退后，普通 benchmark record 仍提供：

- op_name、dtype、mode、level。
- 完整 `shape_detail`。
- Torch latency、Gems latency、speedup。

这些字段构成后处理输入，不再要求 `logical_flops/logical_bytes` 出现在日志中。

## 错误处理

runner：

- NUMA/node/CPU list 不合法：执行前失败。
- 检测到注入或强制 pipeline 环境变量：执行前失败。
- 单算子 pytest 失败：保存 stdout 和状态，继续后续算子。
- timeout：终止该子进程，保存已有输出，继续后续算子。
- 任一算子失败：批次最终返回非零。

映射生成：

- 未知公式、输入类型或固定 workload：不发布映射表。
- 相同 key 在同次生成中得到冲突值：失败。
- 增量更新只覆盖明确选中或日志中出现的 key。

报告：

- malformed JSON、无效 latency、缺映射、重复 key：不发布新顶层结果。
- `error_msg` 非空的 benchmark 行不进入数据表，并写入 coverage。
- 不使用旧日志内可能存在的 `logical_*` 作为默认数据；迁移验证时可用于对比。

## 实施顺序

1. 保存当前两仓状态和相关 commit/file 清单，不触碰用户未提交内容。
2. 在 Agent skill 实现外部公式计算器和固定输入映射表。
3. 使用已有注入日志做 golden parity；确认明确例外只有批准的口径变化。
4. 实现纯日志报告和按算子原子替换。
5. 实现单 job 串行 runner、仓外日志目录和 CPU/memory NUMA 解耦。
6. 使用未注入的普通 benchmark 日志验证完整链路。
7. 定向回退产品仓 `df2dc2603` 的四个文件改动。
8. 更新 `SKILL.md`、`agents/openai.yaml` 和最终命令示例。
9. 完成静态检查、最小真实运行和结果差异审计后，再分别提交两仓改动。

## 验收标准

### Runner

- 任意时刻最多一个 benchmark 子进程。
- CPU node 与 memory/HBM node 可不同。
- record/stdout 全部写入仓外集中目录。
- 只请求 latency_base、latency、speedup。
- 不设置 fusion、pipeline 或 cache 环境变量。
- 不自动 fallback、重试或更改 dtype。
- `run.json` 能完整说明每条日志的运行参数和绑定信息。
- benchmark 完成后自动生成 `<run-dir>/report`；报告失败时 runner 返回非零。

### 映射表

- key 等于规范化的 `op_name + dtype + 完整 shape_detail`。
- 映射表只有 `generated_at` 和 entries，不含 SHA/版本字段。
- 支持全量重建和按日志/算子增量更新。
- 公式计算不创建真实大 Tensor，不运行产品算子。
- 映射缺失时报告严格失败。
- `rwkv_mm_sparsity` 明确使用固定理论 nnz 口径。

### 报告

- 普通无注入日志可直接生成 compute/memory CSV。
- 目标值变化只重算报告，不重跑 benchmark 或重建映射表。
- 更新一个算子后，未选算子结果保持不变。
- 更新失败时旧顶层报告保持不变。
- 报告阶段不 import Torch/FlagGems，不计算公式。
- 现有 CSV 表头和列顺序保持稳定；无法从日志确认的运行事实写 `N/A`。

### 迁移

- `triton-cpu` 不再包含 fusion metric 注入和公式文件。
- 普通 benchmark 性能路径恢复为注入前行为。
- 旧原始日志和报告快照不删除。
- `SKILL.md` 只在新实现通过最小真实验证后切换到新入口。
