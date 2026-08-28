# FlagGems Benchmark Profiling Scripts

## 目录

- [文件组成](#文件组成)
- [使用前检查](#使用前检查)
- [测试流程](#测试流程)
- [实际使用](#实际使用)
- [配置说明](#配置说明)
- [输出结构](#输出结构)
- [TFLOPS 口径](#tflops-口径)
- [统计口径](#统计口径)
- [常见问题](#常见问题)

本目录提供一套围绕 `triton-cpu/FlagGems/benchmark` pytest benchmark 的性能数据采集工具。它通过子进程调用现有 benchmark、`sitecustomize` 注入 Triton 编译 hook、CPU/内存后台采样、`perf stat` 采集 cache/指令/周期计数器，并在运行结束后聚合为 `summary.json` 和 `summary.md`。

脚本默认从自身位置反推工作区：

```text
${AGENT_DIR}/
├── AgentForTritonCPU/
│   └── skills/flaggems-benchmark-profile/
└── triton-cpu/
```

如果 `triton-cpu` 不在默认位置，可以在运行前设置：

```bash
export TRITON_REPO_DIR=/path/to/triton-cpu
```

## 文件组成

- `scripts/run_profile.py`: profiling 主控脚本，读取配置、拼接命令、逐算子执行 pytest、收集产物。
- `scripts/profiling_config.yaml`: 默认配置，包含 tier、系统绑定、profiling 开关和测试列表。
- `scripts/profiling/config.py`: YAML 配置解析和校验。
- `scripts/profiling/compile_hook.py` / `scripts/profiling/sitecustomize.py`: 在 pytest 子进程启动时注入 Triton JIT 编译时间采集。
- `scripts/profiling/monitor.py`: 基于 `psutil` 的 CPU/内存后台采样。
- `scripts/profiling/perf_wrapper.py`: `perf stat` 命令拼接与 CSV 输出解析。
- `scripts/profiling/flops.py`: TFLOPS 公式目录和聚合阶段的 FLOPs/TFLOPS 补齐逻辑。
- `scripts/profiling/system_info.py`: 收集 `lscpu`、`numactl -H`、`free -h`、`uname -a`、`/proc/cpuinfo`。
- `scripts/profiling/result_aggregator.py`: 聚合编译、资源、perf、record log 数据。
- `scripts/tools/perf_heatmap.py`: 从 `perf_report.txt` 或 `perf.data` 生成热点热力图。
- `docs/design.md`: 原始设计文档快照，便于追溯方案来源。

## 使用前检查

默认配置绑定 NUMA node `3` 和 CPU `456-487`，实际使用前请按目标机器调整 `scripts/profiling_config.yaml`：

```yaml
system:
  numa:
    enabled: true
    cpunodebind: 3
    membind: 3
  cpu_affinity: "456-487"
```

依赖检查：

```bash
python3 -c "import yaml; print('yaml OK')"
python3 -c "import psutil; print('psutil OK')"  # optional; /proc fallback is used on Linux
which perf
which numactl
which taskset
```

如果目标环境没有 `perf` 权限，运行时使用 `--skip-perf`。CPU/内存采样优先使用 `psutil`，没有安装时会在 Linux 上回退到 `/proc`。

## 测试流程

以下命令可以在 `AgentForTritonCPU/skills/flaggems-benchmark-profile/` 目录执行。

1. 干跑，确认命令拼接、NUMA/CPU 绑定、输出路径正确：

```bash
python3 scripts/run_profile.py \
  --tests silu_and_mul \
  --record log \
  --dtypes float32 \
  --dry-run
```

2. 基础流程验证，先跳过 `perf`，减少环境权限影响：

```bash
python3 scripts/run_profile.py \
  --tests silu_and_mul \
  --warmup-override 10 \
  --iter-override 10 \
  --record log \
  --dtypes float32 \
  --skip-perf
```

3. 如果 `perf` 可用，再跑完整采集：

```bash
python3 scripts/run_profile.py \
  --tests silu_and_mul \
  --warmup-override 10 \
  --iter-override 10 \
  --record log \
  --dtypes float32
```

4. 逐 shape 模式的小用例验证：

```bash
python3 scripts/run_profile.py \
  --tests silu_and_mul \
  --warmup-override 20 \
  --iter-override 10 \
  --record log \
  --dtypes float32 \
  --run-mode case \
  --case-indices 1
```

`--case-indices 1` 表示只跑 query 返回的第 1 个 shape，适合先验证流程。去掉该参数后会逐个 shape 单独运行并汇总。

5. 检查输出目录，确认以下文件存在：

```bash
ls "$AGENT_DIR"/logs/AgentForTritonCPU/flaggems-benchmark-profile/profile_*/silu_and_mul/
cat "$AGENT_DIR"/logs/AgentForTritonCPU/flaggems-benchmark-profile/profile_*/summary.md
cat "$AGENT_DIR"/logs/AgentForTritonCPU/flaggems-benchmark-profile/profile_*/silu_and_mul/compile.jsonl
cat "$AGENT_DIR"/logs/AgentForTritonCPU/flaggems-benchmark-profile/profile_*/silu_and_mul/cpu_memory.csv
```

基础成功信号：

- `summary.md` 中对应算子状态为 `passed` 或至少有 benchmark record 数据。
- `stdout.txt` 保存了完整 pytest 输出。
- `record.log` 保存现有 `--record log` 的 benchmark JSON 记录。
- `compile.jsonl` 有 Triton JIT 编译记录；若没有触发编译或 hook 未生效，该文件可能为空或不存在。
- 开启 CPU/内存采样时有 `cpu_memory.csv`。
- 开启且环境支持 `perf` 时有 `perf_stat.csv`。

## 实际使用

跑配置中的全部测试：

```bash
python3 scripts/run_profile.py --all
```

默认运行模式是 `group`，即每个算子一个 pytest 子进程：

```bash
python3 scripts/run_profile.py \
  --tests silu_and_mul \
  --warmup-override 20 \
  --iter-override 10 \
  --record log \
  --dtypes float32 \
  --run-mode group
```

`group` 模式下，`compile/cycles/instructions/IPC/cache/CPU/RSS` 是整个算子 benchmark 子进程的统计。它适合看整组成本，但这些数据不能解释为每个 shape 的独立数据。

如果需要每个 shape 独立统计，使用 `case` 模式：

```bash
python3 scripts/run_profile.py \
  --tests silu_and_mul \
  --warmup-override 20 \
  --iter-override 10 \
  --record log \
  --dtypes float32 \
  --run-mode case
```

`case` 模式会先执行一次 benchmark `--query` 获取 shape 列表，然后为每个 shape 生成单独 `shape.yaml`，每个 shape 单独启动 pytest、单独清理独立 Triton cache、单独收集 `compile.jsonl`、`perf_stat.csv`、`cpu_memory.csv` 和 `record.log`。此时 `summary.md` 的 case 行中 `Scope=case`，对应的 `Profile Compile (ms)`、`Cycles`、`Instructions`、`IPC` 等字段才是该 shape 的独立统计。

如果需要查看当前算子/shape 内部最耗时的函数或符号 top10，使用 `perf record` 采样：

```bash
python3 scripts/run_profile.py \
  --tests silu_and_mul \
  --warmup-override 20 \
  --iter-override 10 \
  --record log \
  --dtypes float32 \
  --run-mode case \
  --case-indices 4 \
  --perf-record
```

`--perf-record` 会自动跳过原来的 `perf stat` 计数，避免硬件计数器和采样互相干扰。每个算子或 case 目录会额外落盘：

- `perf.data`: `perf record` 原始采样文件。
- `perf_report.txt`: `perf report --stdio` 完整文本输出。
- `perf_top10.txt`: 从 `perf_report.txt` 提取的前 10 个热点符号行。

如果 `perf.data` 太大或采样开销太高，可以降低采样频率或改 call graph 模式：

```bash
python3 scripts/run_profile.py \
  --tests silu_and_mul \
  --warmup-override 20 \
  --iter-override 10 \
  --record log \
  --dtypes float32 \
  --run-mode case \
  --case-indices 4 \
  --perf-record \
  --perf-record-frequency 49 \
  --perf-record-call-graph fp
```

说明：`perf_top10.txt` 是 Linux perf 的符号级热点，不等同于 Python 函数级 `cProfile`。对 Triton CPU JIT 生成的 `.so`，通常能看到 kernel 符号、`memrefCopy`、`malloc/free`、libm/SLEEF/LLVM runtime 等热点；如果符号不完整，优先看 `perf_report.txt` 和 `perf.data`。

已有 `perf.data` 或 `perf_report.txt` 后，可以生成离线热点热力图：

```bash
python3 scripts/tools/perf_heatmap.py \
  "$AGENT_DIR/logs/AgentForTritonCPU/flaggems-benchmark-profile/<run_name>/silu_and_mul/case_004"
```

脚本会优先复用 case 目录里的 `perf_report.txt`；如果只有 `perf.data`，会自动调用 `perf report --stdio` 生成报告。默认输出：

- `perf_heatmap.html`: 自包含 HTML，可直接用浏览器打开，按 DSO 分组展示热点符号。
- `perf_heatmap.svg`: 可嵌入文档或报告的静态 SVG。

也可以直接传入单个文件，并指定输出路径：

```bash
python3 scripts/tools/perf_heatmap.py \
  /path/to/perf.data \
  --output /tmp/silu_and_mul_heatmap.html \
  --top 80 \
  --refresh-report
```

跑多个指定算子：

```bash
python3 scripts/run_profile.py --tests add mm flash_attention_forward
```

指定输出目录和运行名：

```bash
python3 scripts/run_profile.py \
  --tests add \
  --output-dir "$AGENT_DIR/logs/AgentForTritonCPU/flaggems-benchmark-profile" \
  --run-name add_profile_cpu_node12
```

临时覆盖 warmup/iter：

```bash
python3 scripts/run_profile.py \
  --tests add \
  --warmup-override 100 \
  --iter-override 50
```

限制 dtype：

```bash
python3 scripts/run_profile.py \
  --tests silu_and_mul \
  --record log \
  --dtypes float32
```

跳过 cache metrics 或资源采样：

```bash
python3 scripts/run_profile.py \
  --tests add \
  --skip-perf \
  --skip-cpu-mem
```

并发执行：

```bash
python3 scripts/run_profile.py \
  --tests add mm \
  --jobs 2
```

并发模式会自动禁用 `perf`，避免硬件计数器争用；每个算子会使用独立的 Triton cache 目录。

逐 shape 模式也支持 `--jobs`，但 `--jobs > 1` 时同样会禁用 `perf`。如果重点是 `cycles/IPC/cache miss`，建议保持 `--jobs 1`。

## 配置说明

`tiers` 定义 benchmark 规模，测试项通过 `tier` 引用：

```yaml
tiers:
  light:  { warmup: 50,  iterations: 30 }
  medium: { warmup: 200, iterations: 100 }
  heavy:  { warmup: 500, iterations: 200 }
```

`tests` 定义要跑的 benchmark 文件和 marker：

```yaml
tests:
  - test_file: test_silu_and_mul.py
    marker: silu_and_mul
    tier: medium
```

可对单个测试覆盖配置：

```yaml
tests:
  - test_file: test_mm.py
    marker: mm
    tier: light
    warmup_override: 20
    iter_override: 20
    metrics:
      - latency_base
      - latency
      - speedup
      - tflops
```

注意：`metrics` 默认是 `null`，表示使用现有 benchmark 类自己的默认指标。只有确认该 benchmark 支持某个指标时，才建议显式加入，例如 `tflops`。

## 输出结构

每次运行会生成：

```text
$AGENT_DIR/logs/AgentForTritonCPU/flaggems-benchmark-profile/profile_<timestamp>/
├── config_snapshot.yaml
├── system_info.txt
├── summary.json
├── summary.md
└── <op_name>/
    ├── command.txt
    ├── stdout.txt
    ├── record.log
    ├── compile.jsonl
    ├── cpu_memory.csv
    ├── perf_stat.csv
    ├── perf.data
    ├── perf_report.txt
    ├── perf_top10.txt
    ├── perf_heatmap.html
    └── perf_heatmap.svg
```

`--run-mode case` 时，单个算子目录会多一层：

```text
$AGENT_DIR/logs/AgentForTritonCPU/flaggems-benchmark-profile/profile_<timestamp>/
└── silu_and_mul/
    ├── _query/
    │   ├── command.txt
    │   ├── stdout.txt
    │   └── record.log
    └── case_001/
        ├── shape.yaml
        ├── command.txt
        ├── stdout.txt
        ├── record.log
        ├── compile.jsonl
        ├── cpu_memory.csv
        ├── perf_stat.csv
        ├── perf.data
        ├── perf_report.txt
        ├── perf_top10.txt
        ├── perf_heatmap.html
        └── perf_heatmap.svg
```

`perf.data`、`perf_report.txt` 和 `perf_top10.txt` 只在启用 `--perf-record` 时生成；`perf_heatmap.html` 和 `perf_heatmap.svg` 是后续运行 `scripts/tools/perf_heatmap.py` 后生成。

`summary.json` 适合后续脚本消费，`summary.md` 适合人工查看。

## TFLOPS 口径

`summary.md` 默认只显示一列 `TFLOPS`。该值统一按以下方式得到：

```text
TFLOPS = FLOPs / Gems Latency(ms) * 1e3 / 1e12
```

优先级：

1. 如果 benchmark 的 `record.log` 已经产出 `tflops`，直接使用 benchmark 原生值。
2. 如果 `record.log` 中 `tflops` 为 `null`，聚合阶段会根据 `op_name`、`shape_detail` 和 `latency` 查询静态公式补齐。
3. 如果没有可用公式，`TFLOPS` 保持 `N/A`。

补齐发生在 pytest benchmark 结束后的结果聚合阶段，只做 JSON 解析和纯 Python 整数乘除，不调用算子、不分配大 tensor，不影响 benchmark 计时。

`summary.json` 的每个 case 会额外包含用于追溯的字段：

```json
{
  "tflops": 0.0000202,
  "flops": 16384,
  "tflops_source": "derived",
  "flops_formula_id": "derived:fused_pointwise"
}
```

`tflops_source` 可能是：

- `benchmark`: 来自 FlagGems benchmark 原生 `tflops`。
- `derived`: profiling 聚合阶段按静态公式补齐。
- `none`: 没有原生值，也没有可用公式。

公式详情默认不写入 `summary.md`，需要单独查看：

```bash
python3 -m profiling.flops
python3 -m profiling.flops --json
```

当前内置公式覆盖了以下 benchmark 类型：

- benchmark 原生公式：`mm`、`bmm`、`addmm`、`baddbmm`、`grouped_mm`、unary/binary/scalar pointwise、Tex GLU forward/backward。
- profiling 补齐公式：BLAS-like、conv1d/2d/3d、pooling、reduction/softmax、normalization、attention、fused pointwise。

例如 `silu_and_mul` 当前使用通用 benchmark，原生 `tflops` 为 `null`，聚合阶段会按 `4 * numel` 补齐，最终仍显示在统一的 `TFLOPS` 列。

## 统计口径

- `Profile Wall (s)`: 被 profiling 包裹的 pytest 子进程墙钟时间。`group` 模式是整组 shape；`case` 模式是单个 shape。
- `Profile Compile (ms)`: Triton JIT hook 捕获到的实际编译事件耗时之和。它不是 `wall time - latency`，不会包含 pytest、输入构造、日志、GC、perf 或 Python 调度开销。
- `Torch/Gems Latency (ms)`: FlagGems benchmark 记录的单次/中位 latency，来自现有 benchmark 逻辑。
- `Cycles/Instructions/IPC/Cache miss`: `perf stat` 对被包裹子进程的硬件计数。只有在 `Scope=case` 时才可视为单 shape 数据。
- `TFLOPS`: 理论 FLOPs 除以 Gems latency。优先使用 benchmark 原生值；原生值为空时，聚合阶段使用静态公式补齐。

## 常见问题

- `perf not found in PATH`: 安装系统 perf 工具，或使用 `--skip-perf`。
- `No permission to enable ... event`: 检查 `kernel.perf_event_paranoid` 或使用 `--skip-perf`。
- `numactl: command not found`: 安装 `numactl`，或在配置里设置 `system.numa.enabled: false`。
- `taskset: failed to set pid affinity`: 检查 `system.cpu_affinity` 是否适配目标机器 CPU 编号。
- `requested test(s) not found in config`: 需要先把该算子加入 `scripts/profiling_config.yaml` 的 `tests` 列表。
- `record.log` 不存在：pytest 没有产生 benchmark record，优先查看该算子的 `stdout.txt`。
