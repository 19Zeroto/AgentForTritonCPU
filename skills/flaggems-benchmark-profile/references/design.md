# FlagGems Benchmark 性能数据采集方案

## Contents

- [Context](#context)
- [方案设计](#方案设计)
- [验证方案](#验证方案)
- [Agent 开发指引](#agentsmd--agent-开发指引)

## Context

当前 FlagGems 的 benchmark 框架（`FlagGems/benchmark/`）主要采集算子延迟和 speedup 数据，缺少以下关键性能指标：
- 端到端总耗时
- 算子 JIT 编译耗时
- 运行时 CPU/内存使用率
- L1/L2/L3 cache 使用率及 cache miss 数据

此外，现有框架使用全局统一的 warmup/iter 配置（`conftest.py` 中的 `BenchConfig`），不支持按算子定制。

目标：设计一套低侵入、可移植的 profiling 方案，对所有 benchmark 测试通用，且配置可追溯。

---

## 方案设计

### 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                  scripts/run_profile.py  (新增)                      │
│  主控脚本：读配置 → 设系统环境 → 逐算子启动 pytest + 监控     │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ ResourceMon  │  │ PerfWrapper  │  │ CompileTimer     │  │
│  │ (psutil)     │  │ (perf stat)  │  │ (首次调用计时)    │  │
│  │ CPU/Mem采样  │  │ Cache metrics│  │ per shape+dtype  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  现有的 benchmark 基础设施 (最小化改动)                       │
│  conftest.py / performance_utils.py                         │
│  test_*.py (可选: 提供 kernel 元数据供 perf 关联)             │
│                                                             │
│  采集的三层时间:                                              │
│  ① 编译时间 (新) ② 算子端到端 (现有--mode operator)          │
│  ③ pytest 进程总耗时 (run_profile wall-clock)                │
└─────────────────────────────────────────────────────────────┘
```

### 1. 文件清单（全部新增，零侵入现有代码）

所有文件放在 `FlagGems/benchmark/` 下的新目录中，**不修改任何现有仓库文件**。

```
FlagGems/benchmark/
├── scripts/profiling/                       # 新增目录
│   ├── __init__.py
│   ├── config.py                    # ProfilingConfig dataclass + YAML 解析
│   ├── monitor.py                   # ResourceMonitor (psutil CPU/mem 后台采样)
│   ├── perf_wrapper.py              # perf stat 命令行拼接 + 输出解析
│   ├── compile_hook.py             # sitecustomize: monkey-patch triton.jit 记录编译时间
│   ├── result_aggregator.py         # 所有数据源聚合 → summary.json + summary.md
│   └── system_info.py               # 采集: lscpu, numactl -H, /proc/cpuinfo, free -h, uname
├── scripts/profiling_config.yaml            # 新增: 全局配置 + per-op tier + 系统设置
└── scripts/run_profile.py                   # 新增: 主控脚本，通过 subprocess 调用现有 pytest
```

**与现有框架的交互方式**（全部通过子进程 + 后处理，不修改任何现有文件）：

| 数据项 | 采集方式 | 依赖现有框架 |
|--------|---------|-------------|
| 进程端到端总耗时 | `scripts/run_profile.py` 记录 subprocess wall-clock | 无 |
| 编译耗时 | `compile_hook.py` 作为 sitecustomize 注入，monkey-patch `triton.jit` 记录首次编译时间 | 无 |
| 算子 benchmark 延迟 | 解析 `--record log` 输出的 JSON（`--mode operator` 的算子端到端数据） | 现有 `--mode operator` + `--record log` |
| TFLOPS | 通过 pytest `--metrics tflops` 参数启用，从 `--record log` JSON 解析 | 现有 `BenchmarkMetrics.tflops` 字段 |
| CPU/内存 | `monitor.py` 后台线程采样子进程（通过 `psutil.Process(pid)`） | 无 |
| Cache metrics | `perf stat` 包装整个 pytest 子进程 | 无 |

### 2. 配置设计 (`scripts/profiling_config.yaml`)

核心设计决策（已确认）：
- **环境**: Linux + perf stat
- **编译时间**: 每个 (shape, dtype) 组合单独记录
- **Warmup/Iter**: 定义 3 个测试规模 tier，每个算子引用一个 tier
- **独立使用**: 不依赖 `run_priority_suite.py` 接口

```yaml
# ===== 测试规模 Tier 定义 =====
# 不同算子按计算量分为三档，算子只需引用 tier 名
tiers:
  light:      # 重算子: flash_attention, matmul 等
    warmup: 50
    iterations: 30
  medium:     # 中等算子: silu_and_mul, layer_norm 等 fused ops
    warmup: 200
    iterations: 100
  heavy:      # 轻量算子: add, mul, relu 等 pointwise ops
    warmup: 500
    iterations: 200

# ===== 全局默认 =====
global:
  mode: operator         # operator recommended; kernel retained for compatibility only
  level: core            # core | comprehensive
  default_tier: medium

# ===== 系统配置（完整记录到日志） =====
system:
  # Triton cache 清理（编译时间测量必须清缓存）
  triton_cache_dir: null        # 留空自动读取 $TRITON_CACHE_DIR 环境变量
  clear_triton_cache: true      # 每次运行前 rm -rf $TRITON_CACHE_DIR

  # NUMA 绑定 (通过 numactl)
  numa:
    enabled: true
    cpunodebind: 12             # --cpunodebind=12
    membind: 12                  # --membind=12

  # CPU 核心绑定 (通过 taskset)
  cpu_affinity: "456-487"       # taskset -c 456-487

  # 额外的环境变量
  env_vars:
    OMP_NUM_THREADS: "1"
    MKL_NUM_THREADS: "1"

# ===== Profiling 开关 =====
profiling:
  cpu_memory:
    enabled: true
    interval_sec: 1.0     # psutil 采样间隔，可配
  cache_metrics:
    enabled: true
    backend: perf          # Linux perf stat
    events:                # perf stat -e <events>
      - L1-dcache-loads
      - L1-dcache-load-misses
      - L1-dcache-stores
      - LLC-loads
      - LLC-load-misses
      - LLC-stores
      - dTLB-load-misses
      - branch-misses
      - instructions
      - cycles
  compilation_time:
    enabled: true

# ===== 测试列表 =====
# 每个算子引用一个 tier，必要时可单独覆盖 warmup/iter
tests:
  - test_file: test_silu_and_mul.py
    marker: silu_and_mul
    tier: medium
  - test_file: test_add.py
    marker: add
    tier: heavy
  - test_file: test_mm.py
    marker: mm
    tier: light
  - test_file: test_flash_attention_forward.py
    marker: flash_attention_forward
    tier: light
  # 如需覆盖 tier 的默认值:
  # - test_file: test_silu_and_mul.py
  #   marker: silu_and_mul
  #   tier: medium
  #   warmup_override: 300
  #   iter_override: 150
  # ... 更多
```

### 3. 主控脚本工作流 (`scripts/run_profile.py`)

**单算子执行流程**（对应参考命令的每个环节）：

```
1. 解析命令行参数 + 加载 scripts/profiling_config.yaml
2. 收集系统信息 → system_info.txt (numactl -H, lscpu, /proc/cpuinfo, free -h)
3. 保存 config snapshot → config_snapshot.yaml
4. 逐算子循环:
   a. [可选] clear_triton_cache: 清空 Triton 缓存目录
   b. 启动 ResourceMonitor 后台线程 (psutil)
   c. 拼接完整命令:
      numactl --cpunodebind=X --membind=X \
        taskset -c X-Y \
          perf stat -e <events> -o <perf_out> -x, \
            pytest -s <test_file> -m <marker> \
              --mode <mode> --level <level> \
              --warmup <warmup> --iter <iter> \
              --metrics latency_base latency speedup tflops \
              --record log
   d. subprocess.Popen 执行命令 + 实时 tee stdout
   e. 停止 ResourceMonitor
   f. 收集产物: stdout, record log, cpu_memory.csv, perf_stat.csv
5. 聚合所有算子结果 → summary.json + summary.md
```

**命令拼接示意**（`scripts/run_profile.py` 内部生成的最终命令）：

```bash
# 第一步: 清缓存
rm -rf "${TRITON_CACHE_DIR:-/tmp/triton_cache}"

# 第二步: 主命令（compile_hook 通过 PYTHONPATH 注入）
PYTHONPATH=benchmark/profiling:$PYTHONPATH \
FLAGGEMS_COMPILE_LOG=/path/to/results/silu_and_mul/compile.jsonl \
numactl --cpunodebind=12 --membind=12 \
  taskset -c 456-487 \
    perf stat \
      -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,cycles,instructions \
      -o /path/to/results/silu_and_mul/perf_stat.csv -x, \
      python -m pytest benchmark/test_silu_and_mul.py \
        -s -m silu_and_mul \
        --mode operator --level core \
        --warmup 10 --iter 10 \
        --metrics latency_base latency speedup tflops \
        --record log
```

**命令行接口**:

```bash
# 用配置跑单个测试
python scripts/run_profile.py \
  --config scripts/profiling_config.yaml \
  --tests silu_and_mul

# 跑配置中所有测试
python scripts/run_profile.py --config scripts/profiling_config.yaml --all

# 覆盖 warmup/iter（调试用）
python scripts/run_profile.py \
  --config scripts/profiling_config.yaml \
  --tests silu_and_mul \
  --warmup-override 10 --iter-override 10

# 跳过 perf（环境不支持时）
python scripts/run_profile.py --config scripts/profiling_config.yaml --skip-perf

# 干跑（只打印会生成的命令，不实际执行）
python scripts/run_profile.py --config scripts/profiling_config.yaml --dry-run

### 4. 三个时间层次的定义

| 层次 | 含义 | 测量方式 | 对应关系 |
|------|------|----------|----------|
| **编译耗时** | Triton JIT 将 `@triton.jit` 函数编译为机器码的时间 | 首次调用计时（清除缓存后） | 新采集项 |
| **算子 benchmark 延迟** | Python dispatch + kernel launch + 计算 + sync | `time.time()` 或 benchmark record | 现有 `--mode operator` |
| **进程端到端耗时** | pytest 启动、编译、benchmark 和退出的总耗时 | `scripts/run_profile.py` 的 subprocess wall-clock | profiler 进程计时 |

**标准 benchmark 统一使用 `--mode operator`**。`kernel` 模式仅保留用于兼容旧命令，任何场景都不推荐使用：
- 它把完整 benchmark callable 交给 `do_bench`
- `do_bench` 在每次测量执行前清理 benchmark cache
- 它不是单独 Triton kernel 的入口或纯 kernel 计时方式，因此任何场景都不推荐使用

对于单 kernel 算子（如 `silu_and_mul`, `add`, `relu`）：
- Operator 内部只有一个 `@triton.jit` kernel
- operator 延迟覆盖 Python dispatch、kernel 执行和返回前的同步
- silhouette: `[Python dispatch] [kernel exec] [return]`

**对于多 kernel 算子**（如 `flash_attention_forward`, `cross_entropy_loss`）：
- Operator 内部有多个 `@triton.jit` kernel 串联
- operator 延迟覆盖整条 kernel 链路，是标准 benchmark 的目标指标
- 如需定位 kernel 热点，应保持 `operator` 模式并使用 `perf stat`、`perf record/report`
  或其他 profiling/tracing 工具；不要把 benchmark 的 `kernel` 模式当作单独 kernel 测量

**多 kernel 算子的处理策略**：
- 保持 operator benchmark，使用 `perf record/report` 观察 operator 调用中各 kernel 的符号和热点
- 需要更细粒度时使用专用 profiling/tracing harness，不通过 benchmark 的 `kernel` 模式伪造纯 kernel 数据
- 在 benchmark test 文件中可选地添加 kernel 元数据，帮助关联 profiling 输出中的 kernel 名称

### 5. 编译耗时采集方案 (`scripts/profiling/compile_hook.py`)

**零侵入方案**: monkey-patch `triton.jit` 装饰器，作为 sitecustomize 注入到 pytest 子进程。

**原理**：
- `triton.jit` 是所有 `@triton.jit` kernel 的入口点
- 替换为 wrapper：首次调用时记录时间（含编译），后续调用直接透传
- 编译记录写入 `FLAGGEMS_COMPILE_LOG` 指定的 JSON 文件

**核心代码**（`scripts/profiling/compile_hook.py`，~60 行）：

```python
import time, json, os, triton

_original_jit = triton.jit
_compile_output = os.environ.get("FLAGGEMS_COMPILE_LOG")

class _TimedJIT:
    def __init__(self, fn):
        self._fn = fn
        self._done = False
    def __getattr__(self, name):
        return getattr(self._fn, name)
    def __call__(self, *args, **kwargs):
        if not self._done:
            t0 = time.perf_counter()
            result = self._fn(*args, **kwargs)
            # synchronize 取决于后端（CUDA: torch.cuda.synchronize(), CPU: 无需）
            t1 = time.perf_counter()
            record = {
                "name": getattr(self._fn, '__name__', str(self._fn)),
                "compile_ms": (t1 - t0) * 1000
            }
            if _compile_output:
                with open(_compile_output, 'a') as f:
                    f.write(json.dumps(record) + '\n')
            self._done = True
            return result
        return self._fn(*args, **kwargs)

def _patched_jit(fn=None, **kw):
    if fn is None:
        return lambda f: _patched_jit(f, **kw)
    return _TimedJIT(_original_jit(fn, **kw))

triton.jit = _patched_jit
```

**注入方式**（`scripts/run_profile.py` 自动设置环境变量）：

```bash
PYTHONPATH=benchmark/profiling:$PYTHONPATH \
FLAGGEMS_COMPILE_LOG=/path/to/results/<op>/compile.jsonl \
  pytest ... --mode operator --record log ...
```

**局限性**（需在方案中说明）：
- Monkey-patch 层面的 `_done` flag 意味着同一 `@triton.jit` 函数只记录首次编译。Triton 对不同 shape 可能触发 specialization 重编译，这发生在更底层（cache 机制），`@triton.jit` 层面无法捕获
- 对于 `@triton.autotune` 装饰的 kernel，首次调用可能包含 autotune 过程（远比普通编译耗时），记录中无法区分
- 替代方案: 如果 Triton 的环境变量 `TRITON_PRINT_AUTOTUNING=1` 等能输出编译日志到 stdout，可以从 stdout 解析做补充

### 5. CPU/内存采集方案 (`scripts/profiling/monitor.py`)

```python
class ResourceMonitor:
    """后台线程，定期采样进程 CPU 和内存使用"""
    def __init__(self, interval_sec=0.1):
        self.interval = interval_sec
        self.data: list[dict] = []   # [{ts, cpu_pct, rss_mb, vms_mb}, ...]
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        self._stop.set()
        self._thread.join(timeout=5)
        return self.data
```

- 使用 `psutil.Process().cpu_percent()` 和 `.memory_info()`
- 输出为 CSV（时间, cpu%, rss_mb, vms_mb）
- 跨平台（psutil 支持 Linux/Windows/macOS）

### 6. Cache Metrics 采集方案 (`scripts/profiling/perf_wrapper.py`)

两种方案可选：

**方案 A（推荐）: 外部 `perf stat` 包装**
```bash
perf stat \
  -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,... \
  -o perf_output.txt \
  python -m pytest test_silu_and_mul.py -s --record log ...
```
- 优点: 不需要修改 benchmark 代码，对进程无侵入
- 缺点: Linux only，perf 需安装且有权限
- 解析 `perf stat` 输出（带 `-x,` 输出 CSV 格式易解析）

**方案 B（备选）: Python `perf_event` 绑定**
```python
import perf_event
# 需要额外安装，且 API 较底层
```
- 优点: 可在进程内精确控制测量区间
- 缺点: 依赖 `python-perf-event` 包，兼容性不如外部 perf

**推荐方案 A**，方案 B 作为可选 fallback。

### 7. 输出目录结构

```
benchmark/results/profile_20260623_143052/
├── config_snapshot.yaml       # 完整配置快照
├── system_info.txt            # CPU/OS/驱动/内存信息
├── summary.json               # 聚合的所有指标
├── summary.md                 # 可读的 Markdown 报告
├── silu_and_mul/
│   ├── stdout.txt             # pytest 控制台输出
│   ├── record.log             # benchmark JSON 日志（现有格式）
│   ├── cpu_memory.csv         # CPU/内存时序数据
│   ├── perf_stat.csv          # perf 结果
│   └── compilation.json       # 编译耗时明细
├── add/
│   └── ...
```

### 8. summary.json 格式

```json
{
  "run_info": {
    "generated_at": "2026-06-23T14:30:52",
    "config_file": "scripts/profiling_config.yaml",
    "system": { "cpu_model": "...", "memory_gb": 128, "os": "..." }
  },
  "results": [
    {
      "op_name": "silu_and_mul",
      "test_file": "test_silu_and_mul.py",
      "status": "passed",
      "wall_time_sec": 45.2,
      "compilation": {
        "total_compile_time_ms": 3200.5,
        "per_shape": [
          {"shape": [1024, 1024], "dtype": "float16", "compile_ms": 850.3, "operator_latency_ms": 0.012},
          {"shape": [4096, 4096], "dtype": "float16", "compile_ms": 920.1, "operator_latency_ms": 0.045}
        ]
      },
      "cpu_memory": {
        "peak_cpu_pct": 320.0,
        "avg_cpu_pct": 185.5,
        "peak_rss_mb": 4520,
        "avg_rss_mb": 2800
      },
      "cache": {
        "L1_dcache_load_misses": 1234567,
        "L1_dcache_miss_rate": 0.035,
        "LLC_load_misses": 89012,
        "LLC_miss_rate": 0.12,
        "instructions": 1234567890,
        "cycles": 2345678901,
        "ipc": 0.53
      },
      "tflops": {
        "min": 0.85,
        "max": 12.3,
        "mean": 5.2,
        "per_shape": [
          {"shape": [1024, 1024], "dtype": "float16", "tflops": 10.5},
          {"shape": [4096, 4096], "dtype": "float16", "tflops": 12.3}
        ]
      },
      "benchmark_results": [...]   // 现有的 BenchmarkResult 数组（含 tflops 字段）
    }
  ]
}
```

### 9. 对现有代码的改动

**零改动**。所有 profiling 功能均在新增文件中实现，与现有框架的交互通过：

1. **子进程调用**：`scripts/run_profile.py` 用 `subprocess.Popen` 执行现有 pytest 命令，传递 `--mode`, `--level`, `--warmup`, `--iter`, `--record log` 等现有 CLI 参数
2. **sitecustomize 注入**：`compile_hook.py` 通过 `PYTHONPATH` 注入，monkey-patch `triton.jit` 以记录编译时间（类似 `run_priority_suite.py` 已有的 `_bootstrap/sitecustomize.py` 模式）
3. **后处理解析**：解析 `--record log` 产生的 JSON 日志文件获取 operator 延迟数据
4. **外部监控**：`perf stat` + `psutil` 均在独立进程中运行，对 benchmark 进程无侵入

**为什么选择 monkey-patch 而非修改 `performance_utils.py`**：
- `triton.jit` 是编译的入口点，patch 它可以截获所有 `@triton.jit` 装饰函数的首次编译事件
- 这比修改 benchmark 框架更通用——无论算子如何被调用，编译都会被记录
- 仅需一个很小的 `sitecustomize.py`（~50 行），不触碰仓库任何现有文件

### 10. 决策确认

| 决策项 | 结论 |
|--------|------|
| 运行环境 | **Linux + perf stat**，cache metrics 用外部 perf 包装 |
| 编译时间粒度 | **每个 (shape, dtype) 组合单独记录** |
| Warmup/Iter 配置 | **3 个 tier**（light/medium/heavy），算子引用 tier |
| 接口兼容 | **独立使用**，新 CLI 不依赖 `run_priority_suite.py` |
| CPU 采样间隔 | **默认 1s**，在 config 中可配 |
| perf 范围 | **整个 pytest 进程**，包含 Python 开销；使用 `perf record/report` 可进一步定位其中的 kernel 热点 |

---
## 验证方案

1. 在目标机器上用单个简单 op（如 `test_add.py`）验证完整流程
2. 检查所有输出文件完整性
3. 用 `test_silu_and_mul.py` 验证编译时间采集准确性
4. 用 `perf stat ls` 预验证 perf 可用性

---

## agents.md — Agent 开发指引

> 此文件位于 `FlagGems/benchmark/agents.md`，供 Claude Code agent 实现本方案时参考。

### 一、实现顺序

按依赖关系分 4 个阶段，每个阶段产出可独立验证：

```
Phase 1: 基础设施（无依赖）
  ├── scripts/profiling/__init__.py
  ├── scripts/profiling/config.py           # ProfilingConfig + YAML load
  └── scripts/profiling/system_info.py      # 系统信息采集

Phase 2: 数据采集模块（无互依赖，可并行）
  ├── scripts/profiling/monitor.py           # ResourceMonitor
  ├── scripts/profiling/perf_wrapper.py      # PerfStatRunner
  └── scripts/profiling/compile_hook.py      # sitecustomize

Phase 3: 主控 + 聚合
  ├── scripts/profiling/result_aggregator.py # 汇总所有数据源
  ├── scripts/profiling_config.yaml          # 配置文件
  └── scripts/run_profile.py                 # 主控脚本

Phase 4: 验证
  └── 在目标机器上端到端跑通 test_silu_and_mul.py
```

### 二、各模块接口规范

#### 2.1 `scripts/profiling/config.py`

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SystemConfig:
    triton_cache_dir: Optional[str] = None
    clear_triton_cache: bool = True
    numa_cpunodebind: Optional[int] = None
    numa_membind: Optional[int] = None
    cpu_affinity: Optional[str] = None         # e.g. "456-487"
    env_vars: dict = field(default_factory=dict)

@dataclass
class ProfilingConfig:
    cpu_memory_enabled: bool = True
    cpu_memory_interval_sec: float = 1.0
    cache_metrics_enabled: bool = True
    cache_events: list = field(default_factory=lambda: [
        "L1-dcache-loads", "L1-dcache-load-misses",
        "LLC-loads", "LLC-load-misses", "cycles", "instructions"
    ])
    compilation_time_enabled: bool = True

@dataclass
class TierConfig:
    warmup: int
    iterations: int

@dataclass
class TestEntry:
    test_file: str          # e.g. "test_silu_and_mul.py"
    marker: str             # e.g. "silu_and_mul"
    tier: str = "medium"    # light | medium | heavy
    mode: str = "operator"  # operator recommended; kernel compatibility only
    level: str = "core"

@dataclass
class RunConfig:
    tiers: dict[str, TierConfig]
    global_mode: str
    global_level: str
    default_tier: str
    system: SystemConfig
    profiling: ProfilingConfig
    tests: list[TestEntry]

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        ...
```

**关键行为**:
- `from_yaml()` 加载 YAML 后做校验：tier 名必须在 `tiers` 中存在，test_file 对应的文件存在
- `TestEntry` 的 `mode`/`level` 如果不指定，继承 `global_mode`/`global_level`
- warnup/iter 从 `TestEntry.tier` 查找 `tiers[<name>]` 解析

#### 2.2 `scripts/profiling/system_info.py`

```python
def collect_system_info() -> dict:
    """返回系统信息 dict，写入 system_info.txt"""
    # lscpu / numactl -H / free -h / uname -a / cat /proc/cpuinfo
    ...

def write_system_info(output_dir: str) -> None:
    """将系统信息写入 output_dir/system_info.txt"""
    ...
```

**实现方式**: 用 `subprocess.run()` 执行上述命令，捕获输出拼接。需处理命令不存在的 fallback。

#### 2.3 `scripts/profiling/monitor.py`

```python
class ResourceMonitor:
    def __init__(self, pid: int, interval_sec: float = 1.0):
        ...
    def start(self) -> None:
        """启动后台线程"""
    def stop(self) -> list[dict]:
        """停止并返回 [{ts, cpu_pct, rss_mb, vms_mb}, ...]"""
    def dump_csv(self, path: str) -> None:
        """将采集数据写入 CSV 文件"""
```

**关键行为**:
- `start()` 启动 daemon 线程，循环 `time.sleep(interval_sec)` + `psutil.Process(pid).cpu_percent()` / `.memory_info()`
- `stop()` 设置 `threading.Event()`，join 线程
- `dump_csv()` 首行写 header: `timestamp,cpu_percent,rss_mb,vms_mb`

**依赖**: `psutil`（非标准库，需在环境中安装）

#### 2.4 `scripts/profiling/perf_wrapper.py`

```python
@dataclass
class PerfResult:
    events: dict[str, float]  # e.g. {"L1-dcache-load-misses": 1234567, ...}

class PerfStatRunner:
    def __init__(self, events: list[str], output_csv: str):
        ...

    def build_command(self, target_cmd: list[str]) -> list[str]:
        """拼接 perf stat 命令
        返回: ['perf', 'stat', '-e', 'event1,event2', '-o', out, '-x,', '--', *target_cmd]
        """

    @staticmethod
    def parse_output(csv_path: str) -> PerfResult:
        """解析 perf stat -x, 的 CSV 输出"""
```

**注意**:
- `perf stat` 需要 `--` 分隔 perf 自身的选项和被包装的命令
- `-x,` 让 perf 输出逗号分隔的 CSV，方便解析
- `perf` 可能不在 PATH，需检查 `shutil.which('perf')`

#### 2.5 `scripts/profiling/compile_hook.py`

**这是 sitecustomize 文件，不提供 Python API**。它通过 `PYTHONPATH` 环境变量注入到 pytest 子进程。

行为：
1. `import triton` 后将 `triton.jit` 替换为 wrapper
2. Wrapper 首次调用时记时并写入 `FLAGGEMS_COMPILE_LOG` 环境变量指定的文件
3. 写入 JSONL 格式（每行一个 JSON object）：`{"name": "kernel_name", "compile_ms": 123.4}`

**重要约束**:
- 必须在 `import triton` 后立即 patch，在 flag_gems 导入之前生效
- `PYTHONPATH=benchmark/profiling` 确保 sitecustomize 机制自动导入（Python 启动时自动执行 sitecustomize.py）
- 或者使用 `PYTHONSTARTUP` 环境变量，或者通过 `-c "import compile_hook"` 显式导入

#### 2.6 `scripts/profiling/result_aggregator.py`

```python
class ResultAggregator:
    def __init__(self, run_dir: str):
        ...

    def add_op_result(self, op_name: str,
                      wall_time_sec: float,
                      compile_log_path: Optional[str],
                      cpu_memory_csv: Optional[str],
                      perf_csv: Optional[str],
                      record_log_path: Optional[str],
                      stdout_path: str) -> dict:
        """汇总单个算子的所有数据源，返回统一格式的 dict"""

    def write_summary(self, output_dir: str):
        """写入 summary.json 和 summary.md"""
```

**关键行为**:
- `add_op_result()` 解析各数据源：
  - `compile_log_path` → JSONL → `[{name, compile_ms}, ...]`
  - `cpu_memory_csv` → CSV → `{peak_cpu, avg_cpu, peak_rss, avg_rss}`
  - `perf_csv` → CSV → `{event_name: value, ...}`
  - `record_log_path` → 现有 JSON 格式 → 提取 latency/speedup/tflops 等
    - TFLOPS 注意: 仅 `BlasBenchmark`/`BinaryPointwiseBenchmark`/`UnaryPointwiseBenchmark` 等有实现；其他算子 tflops 可能为 null/0，聚合时需过滤
- `write_summary()` 输出与 plan 中 `summary.json` 格式一致

#### 2.7 `scripts/run_profile.py`

主控脚本，唯一需要命令行接口的模块。

```python
def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument('--config', required=True)
    parser.add_argument('--tests', nargs='*')        # 过滤指定 test
    parser.add_argument('--all', action='store_true') # 跑全部
    parser.add_argument('--output-dir', default='benchmark/results')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--skip-perf', action='store_true')
    parser.add_argument('--skip-cpu-mem', action='store_true')
    parser.add_argument('--warmup-override', type=int)
    parser.add_argument('--iter-override', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    ...
```

**核心流程** (`run_single_op`):

```python
def run_single_op(test: TestEntry, config: RunConfig, run_dir: str) -> dict:
    op_name = test.marker
    op_dir = f"{run_dir}/{op_name}"
    os.makedirs(op_dir, exist_ok=True)

    # 1. 确定 warmup/iter（tier 查找 > override > default）
    tier = config.tiers[test.tier]
    warmup = args.warmup_override or tier.warmup
    iterations = args.iter_override or tier.iterations

    # 2. 构建命令片段
    cmd_parts = []

    # 2a. 环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = f"benchmark/profiling:{env.get('PYTHONPATH', '')}"
    env['FLAGGEMS_COMPILE_LOG'] = f"{op_dir}/compile.jsonl"

    # 2b. numactl
    if config.system.numa_cpunodebind is not None:
        cmd_parts.extend(['numactl',
            f'--cpunodebind={config.system.numa_cpunodebind}',
            f'--membind={config.system.numa_membind}'])

    # 2c. taskset
    if config.system.cpu_affinity:
        cmd_parts.extend(['taskset', '-c', config.system.cpu_affinity])

    # 2d. perf stat
    if config.profiling.cache_metrics_enabled and not args.skip_perf:
        perf_runner = PerfStatRunner(
            events=config.profiling.cache_events,
            output_csv=f"{op_dir}/perf_stat.csv"
        )
        cmd_parts.extend(perf_runner.build_command_prefix())
        # 注意: perf stat 后面跟 -- 然后是被测命令

    # 2e. pytest
    pytest_args = [
        sys.executable, '-m', 'pytest',
        f'benchmark/{test.test_file}',
        '-s', '-m', test.marker,
        f'--mode', test.mode,
        f'--level', test.level,
        f'--warmup', str(warmup),
        f'--iter', str(iterations),
        '--metrics', 'latency_base', 'latency', 'speedup', 'tflops',
        '--record', 'log',
    ]
    cmd_parts.extend(pytest_args)

    # 3. 清缓存（事前）
    if config.system.clear_triton_cache:
        cache_dir = config.system.triton_cache_dir or os.environ.get('TRITON_CACHE_DIR', '/tmp/triton_cache')
        subprocess.run(['rm', '-rf', cache_dir])

    # 4. 启动监控 + 执行
    t_start = time.perf_counter()
    proc = subprocess.Popen(cmd_parts, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    monitor = None
    if config.profiling.cpu_memory_enabled and not args.skip_cpu_mem:
        monitor = ResourceMonitor(proc.pid, config.profiling.cpu_memory_interval_sec)
        monitor.start()

    stdout_text, _ = proc.communicate()  # or stream+tee
    t_end = time.perf_counter()
    wall_time = t_end - t_start

    if monitor:
        monitor.stop()
        monitor.dump_csv(f"{op_dir}/cpu_memory.csv")

    # 5. 收集产物
    with open(f"{op_dir}/stdout.txt", 'w') as f:
        f.write(stdout_text)

    # record log 移动（pytest 在当前目录生成）
    record_log = find_record_log()  # result_*.log
    if record_log:
        shutil.move(record_log, f"{op_dir}/record.log")

    # 6. 返回原始结果（由 ResultAggregator 聚合）
    return {
        'op_name': op_name,
        'wall_time_sec': wall_time,
        'compile_log': f"{op_dir}/compile.jsonl",
        'cpu_memory_csv': f"{op_dir}/cpu_memory.csv",
        'perf_csv': f"{op_dir}/perf_stat.csv",
        'record_log': f"{op_dir}/record.log",
        'stdout': f"{op_dir}/stdout.txt",
    }
```

### 三、YAML 配置文件规范 (`scripts/profiling_config.yaml`)

```yaml
tiers:
  light:    { warmup: 50,  iterations: 30 }
  medium:   { warmup: 200, iterations: 100 }
  heavy:    { warmup: 500, iterations: 200 }

global:
  mode: operator
  level: core
  default_tier: medium

system:
  triton_cache_dir: null          # null = 从环境变量读取
  clear_triton_cache: true
  numa:
    cpunodebind: 12
    membind: 12
  cpu_affinity: "456-487"
  env_vars:
    OMP_NUM_THREADS: "1"
    MKL_NUM_THREADS: "1"

profiling:
  cpu_memory:
    enabled: true
    interval_sec: 1.0
  cache_metrics:
    enabled: true
    events:
      - L1-dcache-loads
      - L1-dcache-load-misses
      - L1-dcache-stores
      - LLC-loads
      - LLC-load-misses
      - LLC-stores
      - dTLB-load-misses
      - branch-misses
      - instructions
      - cycles
  compilation_time:
    enabled: true

tests:
  - test_file: test_silu_and_mul.py
    marker: silu_and_mul
    tier: medium
  - test_file: test_add.py
    marker: add
    tier: heavy
  - test_file: test_mm.py
    marker: mm
    tier: light
```

### 四、依赖清单

| 依赖 | 用途 | 安装方式 |
|------|------|---------|
| `psutil` | CPU/内存采样 | `pip install psutil` |
| `PyYAML` | 配置文件解析 | flag_gems 已有依赖 |
| `perf` | Cache metrics | `apt install linux-tools-common` 或系统自带 |
| `numactl` | NUMA 绑定 | 系统自带（`numactl` 包） |
| `taskset` | CPU affinity | 系统自带（`util-linux` 包） |

### 五、输出文件清单（每次运行生成）

```
benchmark/results/profile_<timestamp>/
├── config_snapshot.yaml       # 完整配置副本
├── system_info.txt            # lscpu, numactl -H, free -h, uname -a
├── summary.json               # 聚合结果（格式见 plan 第 8 节）
├── summary.md                 # Markdown 报告
└── <op_name>/
    ├── stdout.txt             # pytest 完整输出
    ├── record.log             # 现有 --record log 的 JSON 日志
    ├── compile.jsonl          # compile_hook 输出的编译记录
    ├── cpu_memory.csv         # ResourceMonitor 时序数据
    └── perf_stat.csv          # perf stat 输出
```

### 六、关键实现注意事项

1. **compile_hook.py 的注入时机**: 必须确保在 `flag_gems` 导入之前 patch `triton.jit`。`PYTHONPATH` 中的 `sitecustomize.py` 会在 Python 启动时自动执行，早于任何用户代码。如环境不支持 sitecustomize，可改用 `python -c "import compile_hook; import pytest; ..." ` 显式加载。

2. **perf stat 与子进程监控的协调**: `perf stat` 会 fork 一个子进程来运行被测命令。`ResourceMonitor` 需要监控正确的 PID。解决方案：先用 `perf stat ... & echo $!` 获取 perf 进程的 PID，然后通过 `psutil.Process(pid).children()` 找到实际 pytest 进程。

3. **record log 文件位置**: 现有 `conftest.py` 在当前工作目录生成 `result_*.log`。`scripts/run_profile.py` 需在 `cwd=benchmark/` 下运行 pytest，运行后立即 move 到 op 目录。

4. **compile_hook 对 autotune 的处理**: 如果 kernel 使用了 `@triton.autotune`，首次调用包含 autotune 搜索过程，`compile_ms` 会非常大。在 `compile.jsonl` 中，可通过 kernel 名称模式（如包含 `_autotune` 前缀）标记为 autotune 类型。

5. **错误处理**: 每个算子的运行是独立的。单个算子失败不应中断整个 run。`scripts/run_profile.py` 应捕获 `subprocess.CalledProcessError`，记录失败状态，继续下一个算子。

6. **并行执行**: `--jobs N` 使用 `concurrent.futures.ThreadPoolExecutor`（与 `run_priority_suite.py` 一致）。并行时注意：
   - 每个 job 需要独立的 `TRITON_CACHE_DIR`（避免编译缓存冲突）
   - `perf stat` 不支持并行运行（硬件计数器争用）——并行模式下应禁用 perf
   - ResourceMonitor 按 PID 分别监控

### 七、端到端验证步骤

```bash
# 1. 检查依赖
python -c "import psutil, yaml; print('OK')"
which perf && perf stat ls > /dev/null 2>&1 && echo "perf OK"
which numactl && echo "numactl OK"

# 2. 干跑（确认命令拼接正确）
python scripts/run_profile.py \
  --config scripts/profiling_config.yaml \
  --tests silu_and_mul --dry-run

# 3. 实际运行（跳过 perf，先验证基本流程）
python scripts/run_profile.py \
  --config scripts/profiling_config.yaml \
  --tests silu_and_mul --skip-perf

# 4. 检查输出
ls benchmark/results/profile_*/silu_and_mul/
cat benchmark/results/profile_*/silu_and_mul/compile.jsonl
cat benchmark/results/profile_*/silu_and_mul/cpu_memory.csv

# 5. 完整运行（含 perf）
python scripts/run_profile.py \
  --config scripts/profiling_config.yaml \
  --tests silu_and_mul
```
