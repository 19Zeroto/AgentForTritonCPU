# Triton CPU 构建指南

本文面向首次准备工作区的开发者。已经存在可用 LLVM、Python 和 Triton 安装时，
直接使用环境脚本和 rebuild skill，不要重复初始化。

## 工作区布局

```text
$AGENT_DIR/
├── AgentForTritonCPU/
├── triton-cpu/
├── llvm-project/
├── logs/
└── cache/
```

`AGENT_DIR` 默认是 `$HOME/agent`，也可以显式设置。产品源码和 Agent 仓库保持
相邻，不要把 Agent 文件复制进 `triton-cpu`。

## Python 环境

使用 Python 3.11，并按当前 `triton-cpu` checkout 的要求安装 setuptools、wheel、
CMake、Ninja、pybind11、lit、nanobind、NumPy、pytest-xdist 和 PyTorch。环境不在
默认位置时设置 `VENV_DIR`；不要把某个机器的 Conda 路径写进仓库文档。

## LLVM/MLIR

将目标 openEuler LLVM 源码放在 `$AGENT_DIR/llvm-project`，构建并安装到
`$AGENT_DIR/llvm-project/install`。典型配置如下，编译器、分支和镜像按当前机器
调整：

```bash
AGENT_DIR="${AGENT_DIR:-$HOME/agent}"
LLVM_SOURCE_DIR="$AGENT_DIR/llvm-project"
LLVM_BUILD_DIR="$LLVM_SOURCE_DIR/build"
LLVM_INSTALL_DIR="$LLVM_SOURCE_DIR/install"

cmake -G Ninja -S "$LLVM_SOURCE_DIR/llvm" -B "$LLVM_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_DIR" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS='mlir;llvm;clang;clang-tools-extra;lld;compiler-rt;openmp' \
  -DLLVM_TARGETS_TO_BUILD='AArch64;NVPTX;AMDGPU' \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DPython3_EXECUTABLE="$(command -v python3)"
ninja -C "$LLVM_BUILD_DIR"
ninja -C "$LLVM_BUILD_DIR" install
```

编译 LLVM Python binding 和运行 Triton 必须使用同一个 Python 环境。

## Triton CPU

```bash
AGENT_DIR="${AGENT_DIR:-$HOME/agent}"
export LLVM_INSTALL_DIR="$AGENT_DIR/llvm-project/install"
export TRITON_REPO_DIR="$AGENT_DIR/triton-cpu"
source "$AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"
python3 -m pip install --no-build-isolation -e "$TRITON_REPO_DIR/python"
```

环境脚本会导出 LLVM、TritonShared、FlagGems、缓存和共享 CPU backend 路径。
首次安装完成后，后续编译使用：

```bash
bash "$AGENT_DIR/AgentForTritonCPU/skills/triton-cpu-rebuild/scripts/rebuild.sh" --dry-run
```

确认干运行命令无误后再去掉 `--dry-run`。该流程不会自动运行 correctness 或
benchmark；验证请使用测试或性能文档，并显式选择当前机器的线程和 NUMA 绑定。
