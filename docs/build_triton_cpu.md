
# Configuring the Python Environment

First, you need to prepare a Python Environment and install the following dependencies. Python 3.11 is recommanded. Please double-check whether the versions of these dependencies meet the requirements, especially CMake and Ninja.

```bash
pip install "setuptools>=40.8.0"
pip install wheel
pip install "cmake>=3.18,<4.0"
pip install "ninja>=1.11.1"
pip install "pybind11>=2.13.1"
pip install lit
pip install nanobind
pip install numpy
pip install pytest-xdist
pip install torch==2.10.0
```

Because the required AArch64 compiler-rt runtime support must be built with clang, use clang to compile the clang sources themselves. To download clang, you need:

`yum install -y llvm-toolset-17-clang llvm-toolset-17-compiler-rt`

clang17 is installed in '/opt/openEuler/llvm-toolset-17/root/usr/bin/clang'
(920F server on blue zone already installed clang 17 toolset, so just skip this step)

# Building the LLVM Compiler
Activate your Python environment, put the openEuler LLVM 20 source code and build it.
```bash
export AGENT_DIR="${AGENT_DIR:-$HOME/agent}"
cd "$AGENT_DIR"
git clone https://gitcode.com/openeuler/llvm-project.git -b dev_20.1.8 --depth=1
cd llvm-project
mkdir build && cd build
cmake -G Ninja -DCMAKE_C_COMPILER=/opt/openEuler/llvm-toolset-17/root/usr/bin/clang -DCMAKE_CXX_COMPILER=/opt/openEuler/llvm-toolset-17/root/usr/bin/clang++ -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang;clang-tools-extra;lld;compiler-rt;openmp" -DLLVM_TARGETS_TO_BUILD="AArch64;NVPTX;AMDGPU" -DCMAKE_INSTALL_PREFIX=../install -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DPython3_EXECUTABLE=$(which python3) -DMLIR_INCLUDE_INTEGRATION_TESTS=ON -DLLVM_ENABLE_RTTI=ON -DBUILD_SHARED_LIBS=OFF
ninja
ninja install

# Make sure python bindings have been generated.
file tools/mlir/python_packages/mlir_core

# if it does not exist, force build with:
ninja check-mlir-python
```
It is important to note that we are using the MLIR python bindings so the python version used to compile llvm (in this case the one pointed by $(which python3)) must be same we use to run triton.

# Building the Triton Compiler
Pull the Triton-CPU source code and build it.
```bash
cd "$AGENT_DIR"
git clone https://gitcode.com/openeuler/triton-cpu.git
cd triton-cpu
git submodule init
git submodule update
export LLVM_INSTALL_DIR="$AGENT_DIR/llvm-project/install"
export LLVM_INCLUDE_DIRS=$LLVM_INSTALL_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_INSTALL_DIR/lib
export LLVM_SYSPATH=$LLVM_INSTALL_DIR
export PYTHONPATH=$LLVM_INSTALL_DIR/python_packages/mlir_core
export PATH=$LLVM_INSTALL_DIR/bin:$PATH
export TRITON_BUILD_WITH_CLANG_LLD=true
export TRITON_PLUGIN_DIRS=$(pwd)/triton-shared
pip install --no-build-isolation -e python
```

If building stall and shows a time out when downloading dependencies from github.com such as googletests:
	In python/setup.py, force a build option by commenting out the condition.(alternatively, find how to set offline_build)
# Test
Before starting the test, you may need to install some dependencies: pytest-xdist and torch. After the preparation is complete, set the following environment variables:
```bash
export TRITON_DISABLE_LINE_INFO=1
export TRITON_USE_SHARED_BACKEND=1
export LLVM_BINARY_DIR="$AGENT_DIR/llvm-project/install/bin/"
export PYTHONPATH="$AGENT_DIR/llvm-project/install/python_packages/mlir_core"
export TRITON_SHARED_OPT_PATH="$AGENT_DIR/triton-cpu/python/build/cmake.linux-{arch}-cpython-{version}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
```
