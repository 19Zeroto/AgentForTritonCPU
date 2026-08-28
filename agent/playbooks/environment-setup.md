---
name: environment-setup
description: 为 triton-cpu 测试、调试或性能数据收集准备可复现环境
required_context:
  - agent/rules/core.md
  - agent/rules/script-writing.md
  - agent/rules/testing.md
  - agent/rules/reporting.md
  - agent/references/project-map.md
  - agent/playbooks/testing.md
  - skills/environment/SKILL.md
---

# 环境准备 Playbook

## 适用范围

用于固定 Python、LLVM/MLIR、Triton Shared 和 OpenMP 环境。本 Playbook 不分析性能数据，不修改算子或编译器代码。

## 输入

- 用途：正确性测试、失败调试、IR dump 或性能数据收集
- 目标套件或算子
- pytest-xdist worker 数与 `OMP_NUM_THREADS`
- LLVM 安装路径和 `triton-shared-opt` 路径

## 流程

1. 按 `skills/environment/SKILL.md` 检查覆盖变量，再从仓库根目录加载本地环境：

   ```bash
   AGENT_DIR="${AGENT_DIR:-$HOME/agent}"
   source "${AGENT_DIR}/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"
   cd "${AGENT_DIR}/triton-cpu"
   ```

2. 检查 Python 依赖、`llc`、`triton-shared-opt` 和目标 backend 是否可用。
3. 每条测试命令显式设置 `OMP_NUM_THREADS`，且不得超过 32；与 xdist 的组合按 `agent/playbooks/testing.md` 选择。
4. 记录影响复现的环境变量、CPU/NUMA 绑定、工具链路径和已知限制。
5. 不得在未获用户明确授权时修改 CPU 频率、系统服务、NUMA 策略或需要 root 权限的系统配置。

## 环境快照

至少记录：

- 用途和目标套件
- `TRITON_CPU_BACKEND`
- `OMP_NUM_THREADS`
- `LLVM_BINARY_DIR` 和 `TRITON_SHARED_OPT_PATH`
- affinity/NUMA 配置（如有）
- 工具可用性检查结果
- 已知限制和尚未验证项

不在本 Playbook 中自行补全。
