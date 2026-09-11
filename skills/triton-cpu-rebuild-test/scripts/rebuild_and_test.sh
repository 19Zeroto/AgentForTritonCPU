#!/usr/bin/env bash
# 用法：bash <脚本路径> [选项]
# 示例：bash <skill>/scripts/rebuild_and_test.sh --cpu-node 3 --mem-node 3 --cpu-list 456-487
# 设计说明：../references/rebuild-test.md

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_AGENT_DIR="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

AGENT_DIR="${AGENT_DIR:-${DEFAULT_AGENT_DIR}}"
TRITON_REPO_DIR="${TRITON_REPO_DIR:-${AGENT_DIR}/triton-cpu}"
LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-${AGENT_DIR}/llvm-project}"
LLVM_BUILD_DIR_EXPLICIT=0
if [[ -n "${LLVM_BUILD_DIR:-}" ]]; then
  LLVM_BUILD_DIR_EXPLICIT=1
fi
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${LLVM_SOURCE_DIR}/build}"
LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR:-${AGENT_DIR}/llvm-project/install}"
ENV_SCRIPT="${ENV_SCRIPT:-${AGENT_DIR}/AgentForTritonCPU/skills/environment/scripts/triton-cpu-env.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${AGENT_DIR}/logs/AgentForTritonCPU/triton-cpu-rebuild-test}"
CPU_NODE="${DGEGLU_CPU_NODE:-0}"
MEM_NODE="${DGEGLU_MEM_NODE:-${CPU_NODE}}"
CPU_LIST="${DGEGLU_CPU_LIST:-}"
OMP_THREADS="${DGEGLU_OMP_THREADS:-32}"
WARMUP="${DGEGLU_WARMUP:-5}"
ITERATIONS="${DGEGLU_ITERATIONS:-5}"
SKIP_LLVM_UPDATE=0
SKIP_INSTALL=0
BUILD_ONLY=0
DRY_RUN=0
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'EOF'
用法：rebuild_and_test.sh [选项]

更新 LLVM/MLIR，重新安装 triton-cpu，然后只运行 dgeglu float32 benchmark。

选项：
  --repo PATH           Triton CPU 仓库，默认 $AGENT_DIR/triton-cpu
  --llvm-repo PATH      LLVM 源码仓库，默认 $AGENT_DIR/llvm-project
  --llvm-build PATH     LLVM 构建目录，默认 <llvm-repo>/build
  --llvm-install PATH  LLVM 安装目录，默认 $AGENT_DIR/llvm-project/install
  --output-root PATH   外部输出根目录
  --cpu-node SPEC      NUMA CPU node，默认 DGEGLU_CPU_NODE 或 0
  --mem-node SPEC      NUMA memory node，默认 DGEGLU_MEM_NODE 或 CPU node
  --cpu-list SPEC      物理 CPU 范围，例如 456-487；默认由 runner 选择
  --omp-threads N      OpenMP 线程数，默认 32，最大 32
  --warmup N           warmup 次数，默认 5
  --iter N              测量次数，默认 5
  --skip-llvm-update   跳过 git pull、ninja 和 ninja install
  --skip-install       跳过清理 python/build 和 pip editable install
  --build-only         只更新/安装并做 import check，不运行 benchmark
  --dry-run            只打印命令，不拉取、编译、删除、安装或测试
  -h, --help           显示帮助

环境变量覆盖：AGENT_DIR、TRITON_REPO_DIR、LLVM_SOURCE_DIR、LLVM_BUILD_DIR、
LLVM_INSTALL_DIR、ENV_SCRIPT、OUTPUT_ROOT、DGEGLU_CPU_NODE、DGEGLU_MEM_NODE、
DGEGLU_CPU_LIST、DGEGLU_OMP_THREADS、DGEGLU_WARMUP、DGEGLU_ITERATIONS。
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

validate_nonnegative_int() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer: ${value}"
}

make_absolute_from_agent() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$AGENT_DIR" "$path"
  fi
}

while (($# > 0)); do
  case "$1" in
    --repo)
      (($# >= 2)) || die "--repo requires a path"
      TRITON_REPO_DIR="$2"
      shift 2
      ;;
    --llvm-repo)
      (($# >= 2)) || die "--llvm-repo requires a path"
      LLVM_SOURCE_DIR="$2"
      shift 2
      ;;
    --llvm-build)
      (($# >= 2)) || die "--llvm-build requires a path"
      LLVM_BUILD_DIR="$2"
      LLVM_BUILD_DIR_EXPLICIT=1
      shift 2
      ;;
    --llvm-install)
      (($# >= 2)) || die "--llvm-install requires a path"
      LLVM_INSTALL_DIR="$2"
      shift 2
      ;;
    --output-root)
      (($# >= 2)) || die "--output-root requires a path"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --cpu-node)
      (($# >= 2)) || die "--cpu-node requires a NUMA node specification"
      CPU_NODE="$2"
      shift 2
      ;;
    --mem-node)
      (($# >= 2)) || die "--mem-node requires a NUMA memory node specification"
      MEM_NODE="$2"
      shift 2
      ;;
    --cpu-list)
      (($# >= 2)) || die "--cpu-list requires a CPU range"
      CPU_LIST="$2"
      shift 2
      ;;
    --omp-threads)
      (($# >= 2)) || die "--omp-threads requires an integer"
      OMP_THREADS="$2"
      shift 2
      ;;
    --warmup)
      (($# >= 2)) || die "--warmup requires an integer"
      WARMUP="$2"
      shift 2
      ;;
    --iter)
      (($# >= 2)) || die "--iter requires an integer"
      ITERATIONS="$2"
      shift 2
      ;;
    --skip-llvm-update)
      SKIP_LLVM_UPDATE=1
      shift
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --build-only)
      BUILD_ONLY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1 (use --help)"
      ;;
  esac
done

validate_nonnegative_int "--omp-threads" "$OMP_THREADS"
validate_nonnegative_int "--warmup" "$WARMUP"
validate_nonnegative_int "--iter" "$ITERATIONS"
((OMP_THREADS > 0 && OMP_THREADS <= 32)) || die "--omp-threads must be between 1 and 32"
((ITERATIONS > 0)) || die "--iter must be greater than zero"

if ((LLVM_BUILD_DIR_EXPLICIT == 0)); then
  LLVM_BUILD_DIR="${LLVM_SOURCE_DIR}/build"
fi

AGENT_DIR="$(cd -- "$AGENT_DIR" 2>/dev/null && pwd)" || die "AGENT_DIR not found: $AGENT_DIR"
TRITON_REPO_DIR="$(make_absolute_from_agent "$TRITON_REPO_DIR")"
LLVM_SOURCE_DIR="$(make_absolute_from_agent "$LLVM_SOURCE_DIR")"
LLVM_BUILD_DIR="$(make_absolute_from_agent "$LLVM_BUILD_DIR")"
LLVM_INSTALL_DIR="$(make_absolute_from_agent "$LLVM_INSTALL_DIR")"
ENV_SCRIPT="$(make_absolute_from_agent "$ENV_SCRIPT")"
OUTPUT_ROOT="$(make_absolute_from_agent "$OUTPUT_ROOT")"

[[ -d "$TRITON_REPO_DIR" ]] || die "Triton repository not found: $TRITON_REPO_DIR"
[[ -f "$TRITON_REPO_DIR/python/setup.py" ]] || die "not a Triton CPU repository: $TRITON_REPO_DIR"
[[ -d "$LLVM_SOURCE_DIR" ]] || die "LLVM source repository not found: $LLVM_SOURCE_DIR"
[[ -d "$LLVM_BUILD_DIR" ]] || die "LLVM build directory not found: $LLVM_BUILD_DIR"
[[ -f "$LLVM_BUILD_DIR/build.ninja" ]] || die "missing build.ninja: $LLVM_BUILD_DIR"
[[ -d "$LLVM_INSTALL_DIR" ]] || die "LLVM install directory not found: $LLVM_INSTALL_DIR"
[[ -f "$ENV_SCRIPT" ]] || die "environment script not found: $ENV_SCRIPT"

export AGENT_DIR TRITON_REPO_DIR LLVM_SOURCE_DIR LLVM_BUILD_DIR LLVM_INSTALL_DIR

# The shared helper must be sourced so its Conda activation and exports remain
# in this shell for pip, the import check, and the benchmark subprocess.
# shellcheck disable=SC1090
source "$ENV_SCRIPT"
export OMP_NUM_THREADS="$OMP_THREADS"

RUN_ID="$(date +%Y%m%d_%H%M%S_%N)"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
BENCHMARK_SCRIPT="${AGENT_DIR}/AgentForTritonCPU/skills/flaggems-fusion-metrics/scripts/run_fusion_benchmarks.py"
TRITON_PYTHON_BUILD_DIR="${TRITON_REPO_DIR}/python/build"
[[ -f "$BENCHMARK_SCRIPT" ]] || die "benchmark runner not found: $BENCHMARK_SCRIPT"

if ((DRY_RUN == 0)); then
  mkdir -p "$RUN_DIR"
  [[ -w "$RUN_DIR" ]] || die "output directory is not writable: $RUN_DIR"
  {
    echo "AGENT_DIR=$AGENT_DIR"
    echo "TRITON_REPO_DIR=$TRITON_REPO_DIR"
    echo "LLVM_SOURCE_DIR=$LLVM_SOURCE_DIR"
    echo "LLVM_BUILD_DIR=$LLVM_BUILD_DIR"
    echo "LLVM_INSTALL_DIR=$LLVM_INSTALL_DIR"
    echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-<unset>}"
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
    echo "python=$(command -v python3)"
    python3 --version
    echo "OMP_NUM_THREADS=$OMP_THREADS"
    echo "CPU_NODE=$CPU_NODE"
    echo "MEM_NODE=$MEM_NODE"
    echo "CPU_LIST=${CPU_LIST:-<runner-default>}"
  } >"$RUN_DIR/environment.txt"
  {
    printf 'script='
    print_command "${BASH_SOURCE[0]}"
    printf 'args='
    print_command "${ORIGINAL_ARGS[@]}"
  } >"$RUN_DIR/command.txt"
  git -C "$TRITON_REPO_DIR" status --short --branch >"$RUN_DIR/triton-status-before.txt" || true
  git -C "$TRITON_REPO_DIR" rev-parse HEAD >"$RUN_DIR/triton-head-before.txt" || true
  git -C "$LLVM_SOURCE_DIR" status --short --branch >"$RUN_DIR/llvm-status-before.txt" || true
  git -C "$LLVM_SOURCE_DIR" rev-parse HEAD >"$RUN_DIR/llvm-head-before.txt" || true
fi

run_logged() {
  local name="$1"
  shift
  echo
  echo "[$name]"
  print_command "$@"
  if ((DRY_RUN)); then
    return 0
  fi
  set +e
  "$@" 2>&1 | tee "$RUN_DIR/${name}.log"
  local command_status="${PIPESTATUS[0]}"
  set -e
  if ((command_status != 0)); then
    echo "[$name] failed with return code $command_status" >&2
    return "$command_status"
  fi
}

run_cwd_logged() {
  local name="$1"
  local workdir="$2"
  shift 2
  echo
  echo "[$name] (cwd=$workdir)"
  print_command "$@"
  if ((DRY_RUN)); then
    return 0
  fi
  set +e
  (
    cd "$workdir" && "$@"
  ) 2>&1 | tee "$RUN_DIR/${name}.log"
  local command_status="${PIPESTATUS[0]}"
  set -e
  if ((command_status != 0)); then
    echo "[$name] failed with return code $command_status" >&2
    return "$command_status"
  fi
}

if ((SKIP_LLVM_UPDATE == 0)); then
  if ((DRY_RUN == 0)); then
    llvm_dirty="$(git -C "$LLVM_SOURCE_DIR" status --porcelain)"
    [[ -z "$llvm_dirty" ]] || die "LLVM worktree is dirty; commit or stash it before git pull"
  fi
  run_cwd_logged llvm-pull "$LLVM_SOURCE_DIR" git pull --ff-only
  run_cwd_logged llvm-build "$LLVM_BUILD_DIR" ninja
  run_cwd_logged llvm-install "$LLVM_BUILD_DIR" ninja install
else
  echo "[llvm] update/build/install skipped by --skip-llvm-update"
fi

if ((SKIP_INSTALL == 0)); then
  [[ "$TRITON_PYTHON_BUILD_DIR" == "$TRITON_REPO_DIR/python/build" ]] || \
    die "refusing to clean unexpected path: $TRITON_PYTHON_BUILD_DIR"
  run_logged clean-triton-build rm -rf -- "$TRITON_PYTHON_BUILD_DIR"
  run_cwd_logged install "$TRITON_REPO_DIR" python3 -m pip install --no-build-isolation -e python
  # Refresh the generated triton-shared-opt path and external cache after the
  # old python/build tree has been removed and recreated.
  source "$ENV_SCRIPT"
  export OMP_NUM_THREADS="$OMP_THREADS"
else
  echo "[triton] clean/install skipped by --skip-install"
fi

run_logged import-check python3 -c \
  'import torch, triton; print("triton:", triton.__file__); print("torch:", torch.__version__)'

if ((BUILD_ONLY)); then
  echo "Build-only run complete: $RUN_DIR"
  exit 0
fi

benchmark_command=(
  python3 "$BENCHMARK_SCRIPT"
  --ops dgeglu
  --mode operator
  --level core
  --warmup "$WARMUP"
  --iter "$ITERATIONS"
  --dtypes float32
  --omp-threads "$OMP_THREADS"
  --cpu-node "$CPU_NODE"
  --mem-node "$MEM_NODE"
  --output-dir "$RUN_DIR/benchmark"
)
if [[ -n "$CPU_LIST" ]]; then
  benchmark_command+=(--cpu-list "$CPU_LIST")
fi

run_logged benchmark "${benchmark_command[@]}"
if ((DRY_RUN)); then
  echo "Dry-run complete; no build, install, deletion, benchmark, or result files were performed: $RUN_DIR"
else
  echo "Rebuild and dgeglu float32 benchmark complete: $RUN_DIR"
fi
