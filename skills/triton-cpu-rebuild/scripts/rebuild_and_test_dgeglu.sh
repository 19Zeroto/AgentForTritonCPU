#!/usr/bin/env bash
# 用法：bash <skill>/scripts/rebuild_and_test_dgeglu.sh [选项]
# 示例：bash <skill>/scripts/rebuild_and_test_dgeglu.sh --cpu-node <node> --mem-node <node> --dry-run
# 设计说明：../references/dgeglu-validation.md

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENTFORTRITONCPU_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
AGENT_DIR="${AGENT_DIR:-${HOME}/agent}"
REBUILD_SCRIPT="$SCRIPT_DIR/rebuild.sh"
ENV_SCRIPT="${ENV_SCRIPT:-${AGENTFORTRITONCPU_DIR}/skills/environment/scripts/triton-cpu-env.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${AGENT_DIR}/logs/AgentForTritonCPU/dgeglu-rebuild-validation}"
CPU_NODE=""
MEM_NODE=""
CPU_LIST=""
OMP_THREADS=32
WARMUP=5
ITERATIONS=5
SKIP_REBUILD=0
SKIP_LLVM_UPDATE=0
SKIP_TRITON_INSTALL=0
DRY_RUN=0

usage() {
  cat <<'EOF'
用法：rebuild_and_test_dgeglu.sh [选项]

先调用通用 rebuild，再直接运行主仓 FlagGems benchmark 的 dgeglu float32
case。指标由 benchmark 自身的 get_gbps() 产生；本脚本不维护公式或报告转换。
CPU 和 memory node 必须按当前机器显式给出。

选项：
  --cpu-node SPEC           NUMA CPU node（benchmark 必需）
  --mem-node SPEC           NUMA memory node（benchmark 必需）
  --cpu-list SPEC           可选物理 CPU 范围；默认由 CPU node 决定
  --omp-threads N           OpenMP 线程数，默认 32，最大 32
  --warmup N                warmup 次数，默认 5
  --iter N                  测量次数，默认 5
  --output-root PATH        仓外输出根目录
  --env-script PATH         环境加载脚本
  --skip-rebuild            复用现有安装，只运行 import check 和 benchmark
  --skip-llvm-update        rebuild 时跳过 LLVM 更新/安装
  --skip-triton-install     rebuild 时跳过 Triton 清理/安装
  --dry-run                 只检查并打印两个阶段的命令
  -h, --help                显示帮助

需要让构建和 benchmark 使用同一 CPU/NUMA policy 时，在调用本脚本的外层使用
numactl/taskset；不要把文档中的示例节点当成当前机器配置。
EOF
}

die() {
  echo "ERROR: $*"
  exit 1
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

while (($# > 0)); do
  case "$1" in
    --cpu-node)
      (($# >= 2)) || die "--cpu-node requires a value"
      CPU_NODE="$2"
      shift 2
      ;;
    --mem-node)
      (($# >= 2)) || die "--mem-node requires a value"
      MEM_NODE="$2"
      shift 2
      ;;
    --cpu-list)
      (($# >= 2)) || die "--cpu-list requires a value"
      CPU_LIST="$2"
      shift 2
      ;;
    --omp-threads)
      (($# >= 2)) || die "--omp-threads requires a value"
      OMP_THREADS="$2"
      shift 2
      ;;
    --warmup)
      (($# >= 2)) || die "--warmup requires a value"
      WARMUP="$2"
      shift 2
      ;;
    --iter)
      (($# >= 2)) || die "--iter requires a value"
      ITERATIONS="$2"
      shift 2
      ;;
    --output-root)
      (($# >= 2)) || die "--output-root requires a path"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --env-script)
      (($# >= 2)) || die "--env-script requires a path"
      ENV_SCRIPT="$2"
      shift 2
      ;;
    --skip-rebuild)
      SKIP_REBUILD=1
      shift
      ;;
    --skip-llvm-update)
      SKIP_LLVM_UPDATE=1
      shift
      ;;
    --skip-triton-install)
      SKIP_TRITON_INSTALL=1
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

[[ -n "$CPU_NODE" ]] || die "--cpu-node is required"
[[ -n "$MEM_NODE" ]] || die "--mem-node is required"
[[ "$OMP_THREADS" =~ ^[0-9]+$ ]] || die "--omp-threads must be an integer"
((OMP_THREADS > 0 && OMP_THREADS <= 32)) || \
  die "--omp-threads must be between 1 and 32"
[[ "$WARMUP" =~ ^[0-9]+$ ]] || die "--warmup must be a non-negative integer"
[[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]] || die "--iter must be greater than zero"
[[ -x "$REBUILD_SCRIPT" ]] || die "rebuild script not executable: $REBUILD_SCRIPT"
[[ -f "$ENV_SCRIPT" ]] || die "environment script not found: $ENV_SCRIPT"
command -v numactl >/dev/null 2>&1 || die "numactl is required for NUMA binding"

# The generic rebuild runs in a child process, so its sourced environment does
# not propagate here. Load it again before invoking the main-repo benchmark.
# shellcheck disable=SC1090
source "$ENV_SCRIPT"
BENCHMARK_FILE="$TRITON_REPO_DIR/FlagGems/benchmark/test_dgeglu.py"
[[ -f "$BENCHMARK_FILE" ]] || die "benchmark file not found: $BENCHMARK_FILE"

RUN_ID="$(date +%Y%m%d_%H%M%S_%N)"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
if ((DRY_RUN == 0)); then
  mkdir -p "$RUN_DIR"
fi

if ((SKIP_REBUILD == 0)); then
  rebuild_args=(--output-root "$RUN_DIR/rebuild" --env-script "$ENV_SCRIPT")
  ((SKIP_LLVM_UPDATE)) && rebuild_args+=(--skip-llvm-update)
  ((SKIP_TRITON_INSTALL)) && rebuild_args+=(--skip-triton-install)
  ((DRY_RUN)) && rebuild_args+=(--dry-run)
  bash "$REBUILD_SCRIPT" "${rebuild_args[@]}"
else
  echo "[rebuild] skipped"
  if ((DRY_RUN == 0)); then
    python3 -c 'import torch, triton; print("triton:", triton.__file__); print("torch:", torch.__version__)' \
      2>&1 | tee "$RUN_DIR/import-check.log"
  fi
fi

benchmark_cmd=(
  numactl
  "--cpunodebind=$CPU_NODE"
  "--membind=$MEM_NODE"
)
[[ -n "$CPU_LIST" ]] && benchmark_cmd+=("--physcpubind=$CPU_LIST")
benchmark_cmd+=(
  python3 -m pytest "$BENCHMARK_FILE"
  -q --tb=no -m dgeglu
  --mode operator --level core
  --warmup "$WARMUP" --iter "$ITERATIONS"
  --dtypes float32 --record log
)

echo "[benchmark] main-repo dgeglu get_gbps()"
print_command "OMP_NUM_THREADS=$OMP_THREADS" "MKL_NUM_THREADS=$OMP_THREADS" \
  "PYTHONPATH=$TRITON_REPO_DIR/FlagGems/benchmark" "${benchmark_cmd[@]}"
if ((DRY_RUN)); then
  echo "Dry-run complete; no rebuild or benchmark was performed."
  exit 0
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/triton-cpu-dgeglu.XXXXXX")"
cleanup() {
  rm -rf -- "$WORK_DIR"
}
trap cleanup EXIT

set +e
(
  cd "$WORK_DIR" || exit 1
  export OMP_NUM_THREADS="$OMP_THREADS"
  export MKL_NUM_THREADS="$OMP_THREADS"
  export PYTHONPATH="$TRITON_REPO_DIR/FlagGems/benchmark${PYTHONPATH:+:$PYTHONPATH}"
  "${benchmark_cmd[@]}"
) 2>&1 | tee "$RUN_DIR/benchmark.log"
status="${PIPESTATUS[0]}"
set -e

for record in "$WORK_DIR"/*.log; do
  [[ -e "$record" ]] || continue
  cp -- "$record" "$RUN_DIR/$(basename "$record")"
done
printf '%s\n' "$status" >"$RUN_DIR/benchmark.exit_code"
((status == 0)) || die "dgeglu benchmark failed; see $RUN_DIR/benchmark.log"
echo "dgeglu rebuild validation complete: $RUN_DIR"

