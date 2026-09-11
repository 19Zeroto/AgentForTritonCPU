---
name: testing
description: 选择并运行 Triton CPU 或 FlagGems 的定向正确性测试与调试命令
required_context:
  - agent/rules/script-writing.md
  - agent/rules/testing.md
  - agent/rules/reporting.md
  - agent/references/project-map.md
---

# Testing Playbook

批量、并行、断点续跑和结果汇总使用 `skills/test-suite/SKILL.md`。性能 benchmark
使用 `agent/references/benchmark-system.md` 和对应性能 skill；本 Playbook 只保留
定向正确性验证与调试入口。

## Setup

从产品仓根运行，并显式设置线程数：

```bash
AGENT_DIR="${AGENT_DIR:-$HOME/agent}"
source "$AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"
cd "$TRITON_REPO_DIR"
```

## Triton `test_core.py`

完整文件可使用 xdist：

```bash
OMP_NUM_THREADS=32 python3 -m pytest -q --tb=no -n32 --device=cpu \
  python/test/unit/language/test_core.py -m cpu
```

一个或少量 case 使用单进程；需要查看 dump 或 traceback 时才启用 `-s -v`：

```bash
OMP_NUM_THREADS=32 python3 -m pytest -q --tb=no --device=cpu \
  'python/test/unit/language/test_core.py::<case>' -m cpu

OMP_NUM_THREADS=32 python3 -m pytest -s -v --device=cpu \
  'python/test/unit/language/test_core.py::<case>' -m cpu
```

## FlagGems correctness

`FlagGems/tests` 不使用 `test_core.py` 的 CPU marker。根据机器资源选择 xdist 与
OpenMP 组合，总并发不得超过 `agent/rules/testing.md` 的上限：

```bash
OMP_NUM_THREADS=16 python3 -m pytest -q --tb=no -n16 \
  FlagGems/tests/<test_file.py>

OMP_NUM_THREADS=32 python3 -m pytest -s -v \
  'FlagGems/tests/<test_file.py>::<case>'
```

## Report

- pytest 返回零只证明所选目标完成，不代表全量通过。
- `SKIPPED` 不计作实际执行覆盖。
- 编译、lowering 或运行时失败不是有效的正确性或性能样本。
- 报告完整命令、环境、目标 case、实际计数和第一个根因错误。
