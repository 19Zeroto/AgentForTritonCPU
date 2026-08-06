#!/usr/bin/env bash

# Usage:
#   source "$AGENT_DIR/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced, not executed." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_AGENTFORTRITONCPU_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export AGENTFORTRITONCPU_DIR="${AGENTFORTRITONCPU_DIR:-$DEFAULT_AGENTFORTRITONCPU_DIR}"
DEFAULT_AGENT_DIR="$(cd "$AGENTFORTRITONCPU_DIR/.." && pwd)"
export AGENT_DIR="${AGENT_DIR:-$DEFAULT_AGENT_DIR}"

_find_conda_base() {
  if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    "$CONDA_EXE" info --base 2>/dev/null
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    conda info --base 2>/dev/null
    return
  fi

  local candidate
  for candidate in \
    "$AGENT_DIR/miniconda3" \
    "$AGENT_DIR/anaconda3" \
    "$HOME/miniconda3" \
    "$HOME/anaconda3"; do
    if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  return 1
}

_activate_conda_env() {
  local conda_base env_name
  env_name="triton-yzc"
  conda_base="$(_find_conda_base)" || return 1
  if [[ ! -f "$conda_base/etc/profile.d/conda.sh" ]]; then
    return 1
  fi

  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$env_name" >/dev/null 2>&1 || return 1
}

if ! _activate_conda_env; then
  echo "Conda env 'triton-yzc' not activated; falling back to Python venv" >&2
  if [[ -n "${VENV_DIR:-}" ]]; then
    export VENV_DIR
  elif [[ -f "$AGENT_DIR/myenv/bin/activate" ]]; then
    export VENV_DIR="$AGENT_DIR/myenv"
  elif [[ -f "$AGENT_DIR/../myenv/bin/activate" ]]; then
    export VENV_DIR="$(cd "$AGENT_DIR/.." && pwd)/myenv"
  else
    echo "Cannot find Python venv. Set VENV_DIR or create myenv next to/in AGENT_DIR." >&2
    return 1
  fi

  if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Missing venv activate: $VENV_DIR/bin/activate" >&2
    return 1
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi
unset -f _find_conda_base _activate_conda_env

export LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR:-$AGENT_DIR/llvm-project/install}"
export TRITON_REPO_DIR="${TRITON_REPO_DIR:-$AGENT_DIR/triton-cpu}"

if [[ ! -x "$LLVM_INSTALL_DIR/bin/mlir-opt" ]]; then
  echo "Missing mlir-opt: $LLVM_INSTALL_DIR/bin/mlir-opt" >&2
  return 1
fi
if [[ ! -d "$LLVM_INSTALL_DIR/python_packages/mlir_core" ]]; then
  echo "Missing MLIR Python package dir: $LLVM_INSTALL_DIR/python_packages/mlir_core" >&2
  return 1
fi
if [[ ! -d "$TRITON_REPO_DIR" ]]; then
  echo "Missing Triton repo dir: $TRITON_REPO_DIR" >&2
  return 1
fi

export LLVM_SYSPATH="$LLVM_INSTALL_DIR"
export LLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm"
export MLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir"
export LLVM_INCLUDE_DIRS="$LLVM_INSTALL_DIR/include"
export MLIR_INCLUDE_DIRS="$LLVM_INSTALL_DIR/include"
export LLVM_LIBRARY_DIR="$LLVM_INSTALL_DIR/lib"
export LLVM_BINARY_DIR="$LLVM_INSTALL_DIR/bin/"

export CMAKE_PREFIX_PATH="$LLVM_INSTALL_DIR${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export PYTHONPATH="$LLVM_INSTALL_DIR/python_packages/mlir_core${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$LLVM_INSTALL_DIR/bin:$PATH"

export TRITON_BUILD_WITH_CLANG_LLD=true
export TRITON_PLUGIN_DIRS="$TRITON_REPO_DIR/triton-shared"
unset TRITON_OFFLINE_BUILD

TRITON_SHARED_OPT_PATH_FOUND="$(find "$TRITON_REPO_DIR/python/build" -path '*triton-shared-opt' -type f 2>/dev/null | head -n 1 || true)"
if [[ -n "$TRITON_SHARED_OPT_PATH_FOUND" ]]; then
  export TRITON_SHARED_OPT_PATH="$TRITON_SHARED_OPT_PATH_FOUND"
fi

export TRITON_DISABLE_LINE_INFO=1
export TRITON_USE_SHARED_BACKEND=1
export GEMS_VENDOR=kunpeng
export TRITON_CACHE_DIR="$AGENT_DIR/cache/tmp-$(date +%Y%m%d-%H%M%S)-$$/"

echo "Triton-CPU environment loaded"
echo "AGENT_DIR=$AGENT_DIR"
echo "AGENTFORTRITONCPU_DIR=$AGENTFORTRITONCPU_DIR"
echo "VENV_DIR=${VENV_DIR:-<not used>}"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<not used>}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-<not used>}"
echo "python=$(command -v python3)"
python3 --version
echo "LLVM_INSTALL_DIR=$LLVM_INSTALL_DIR"
echo "TRITON_REPO_DIR=$TRITON_REPO_DIR"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "TRITON_SHARED_OPT_PATH=${TRITON_SHARED_OPT_PATH:-<not found>}"
