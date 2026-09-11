# Bootstrap the Triton CPU toolchain

Use this guide only for an initial workspace setup. Routine activation belongs
to `scripts/triton-cpu-env.sh`; rebuilds of an existing installation belong to
the `triton-cpu-rebuild` skill.

## Python environment

Create a Python 3.11 environment and install the versions required by the
current `triton-cpu` checkout. At minimum the build uses setuptools, wheel,
CMake, Ninja, pybind11, lit, nanobind, NumPy, pytest-xdist, and PyTorch. Prefer
the product repository's pinned requirements when they differ from this list.

Set `VENV_DIR` when the environment is not named `triton-yzc` in Conda and is
not located at `$AGENT_DIR/myenv`.

## LLVM/MLIR

Clone the intended openEuler LLVM branch beside the repositories, configure an
AArch64 build, and install it under `$AGENT_DIR/llvm-project/install`. A typical
configuration is:

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

Compiler paths, source branch, dependency mirrors, and offline build options
are host/site choices; do not encode a particular server in shared rules.

## Triton CPU

After cloning `triton-cpu` and initializing its submodules:

```bash
AGENT_DIR="${AGENT_DIR:-$HOME/agent}"
export LLVM_INSTALL_DIR="$AGENT_DIR/llvm-project/install"
export TRITON_REPO_DIR="$AGENT_DIR/triton-cpu"
source "$AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"
python3 -m pip install --no-build-isolation -e "$TRITON_REPO_DIR/python"
```

The Python used to build LLVM's MLIR bindings must match the environment used
to import Triton.

## Verification

Source the environment helper and verify `python3`, `mlir-opt`,
`triton-shared-opt`, `triton.__file__`, and the selected backend. Do not claim
the environment is ready when a required path is missing.
