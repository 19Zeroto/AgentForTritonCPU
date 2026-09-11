#!/usr/bin/env bash
# 用法：bash <skill>/scripts/rebuild.sh [选项]
# 示例：bash <skill>/scripts/rebuild.sh --dry-run
# 设计说明：../references/workflow.md

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENTFORTRITONCPU_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
AGENT_DIR="${AGENT_DIR:-${HOME}/agent}"
TRITON_REPO_DIR="${TRITON_REPO_DIR:-${AGENT_DIR}/triton-cpu}"
LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-${AGENT_DIR}/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${LLVM_SOURCE_DIR}/build}"
LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR:-${AGENT_DIR}/llvm-project/install}"
ENV_SCRIPT="${ENV_SCRIPT:-${AGENTFORTRITONCPU_DIR}/skills/environment/scripts/triton-cpu-env.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${AGENT_DIR}/logs/AgentForTritonCPU/triton-cpu-rebuild}"
SKIP_LLVM_UPDATE=0
SKIP_TRITON_INSTALL=0
DRY_RUN=0
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'EOF'
用法：rebuild.sh [选项]

更新并安装现有 LLVM/MLIR 构建树，清理 Triton 的生成 Python build 目录，
重新执行 editable install，最后验证导入路径。本脚本不运行 correctness 或
benchmark；验证由调用者选择对应 playbook/skill。

选项：
  --repo PATH              Triton CPU 仓库，默认 $AGENT_DIR/triton-cpu
  --llvm-repo PATH         LLVM 源码仓库，默认 $AGENT_DIR/llvm-project
  --llvm-build PATH        LLVM 构建目录，默认 <llvm-repo>/build
  --llvm-install PATH      LLVM 安装目录，默认 $AGENT_DIR/llvm-project/install
  --env-script PATH        环境加载脚本
  --output-root PATH       仓外日志根目录
  --skip-llvm-update       跳过 LLVM git pull、ninja 和 ninja install
  --skip-triton-install    跳过清理 python/build 和 editable install
  --dry-run                只打印命令，不拉取、编译、删除或安装
  -h, --help               显示帮助

环境变量覆盖：AGENT_DIR、TRITON_REPO_DIR、LLVM_SOURCE_DIR、LLVM_BUILD_DIR、
LLVM_INSTALL_DIR、ENV_SCRIPT、OUTPUT_ROOT。
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

absolute_from_agent() {
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
      shift 2
      ;;
    --llvm-install)
      (($# >= 2)) || die "--llvm-install requires a path"
      LLVM_INSTALL_DIR="$2"
      shift 2
      ;;
    --env-script)
      (($# >= 2)) || die "--env-script requires a path"
      ENV_SCRIPT="$2"
      shift 2
      ;;
    --output-root)
      (($# >= 2)) || die "--output-root requires a path"
      OUTPUT_ROOT="$2"
      shift 2
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

AGENT_DIR="$(cd -- "$AGENT_DIR" 2>/dev/null && pwd)" || \
  die "AGENT_DIR not found: $AGENT_DIR"
TRITON_REPO_DIR="$(absolute_from_agent "$TRITON_REPO_DIR")"
LLVM_SOURCE_DIR="$(absolute_from_agent "$LLVM_SOURCE_DIR")"
LLVM_BUILD_DIR="$(absolute_from_agent "$LLVM_BUILD_DIR")"
LLVM_INSTALL_DIR="$(absolute_from_agent "$LLVM_INSTALL_DIR")"
ENV_SCRIPT="$(absolute_from_agent "$ENV_SCRIPT")"
OUTPUT_ROOT="$(absolute_from_agent "$OUTPUT_ROOT")"

[[ -f "$TRITON_REPO_DIR/python/setup.py" ]] || \
  die "not a Triton CPU repository: $TRITON_REPO_DIR"
git -C "$LLVM_SOURCE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
  die "LLVM Git repository not found: $LLVM_SOURCE_DIR"
[[ -f "$LLVM_BUILD_DIR/build.ninja" ]] || die "missing build.ninja: $LLVM_BUILD_DIR"
[[ -d "$LLVM_INSTALL_DIR" ]] || die "LLVM install directory not found: $LLVM_INSTALL_DIR"
[[ -f "$ENV_SCRIPT" ]] || die "environment script not found: $ENV_SCRIPT"

export AGENT_DIR TRITON_REPO_DIR LLVM_SOURCE_DIR LLVM_BUILD_DIR LLVM_INSTALL_DIR
# shellcheck disable=SC1090
source "$ENV_SCRIPT"

RUN_ID="$(date +%Y%m%d_%H%M%S_%N)"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
TRITON_PYTHON_BUILD_DIR="$TRITON_REPO_DIR/python/build"

if ((DRY_RUN == 0)); then
  mkdir -p "$RUN_DIR"
  [[ -w "$RUN_DIR" ]] || die "output directory is not writable: $RUN_DIR"
  {
    echo "AGENT_DIR=$AGENT_DIR"
    echo "AGENTFORTRITONCPU_DIR=$AGENTFORTRITONCPU_DIR"
    echo "TRITON_REPO_DIR=$TRITON_REPO_DIR"
    echo "LLVM_SOURCE_DIR=$LLVM_SOURCE_DIR"
    echo "LLVM_BUILD_DIR=$LLVM_BUILD_DIR"
    echo "LLVM_INSTALL_DIR=$LLVM_INSTALL_DIR"
    echo "python=$(command -v python3)"
    python3 --version
  } >"$RUN_DIR/environment.txt"
  {
    printf 'script='
    print_command "${BASH_SOURCE[0]}"
    printf 'args='
    print_command "${ORIGINAL_ARGS[@]}"
  } >"$RUN_DIR/command.txt"
  git -C "$TRITON_REPO_DIR" status --short --branch >"$RUN_DIR/triton-status-before.txt" || true
  git -C "$LLVM_SOURCE_DIR" status --short --branch >"$RUN_DIR/llvm-status-before.txt" || true
fi

run_logged() {
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
  local status="${PIPESTATUS[0]}"
  set -e
  ((status == 0)) || return "$status"
}

if ((SKIP_LLVM_UPDATE == 0)); then
  if ((DRY_RUN == 0)); then
    [[ -z "$(git -C "$LLVM_SOURCE_DIR" status --porcelain)" ]] || \
      die "LLVM worktree is dirty; commit or stash it before git pull"
  fi
  run_logged llvm-pull "$LLVM_SOURCE_DIR" git pull --ff-only
  run_logged llvm-build "$LLVM_BUILD_DIR" ninja
  run_logged llvm-install "$LLVM_BUILD_DIR" ninja install
else
  echo "[llvm] update/build/install skipped"
fi

if ((SKIP_TRITON_INSTALL == 0)); then
  [[ "$TRITON_PYTHON_BUILD_DIR" == "$TRITON_REPO_DIR/python/build" ]] || \
    die "refusing to clean unexpected path: $TRITON_PYTHON_BUILD_DIR"
  echo
  echo "[clean-triton-build]"
  print_command rm -rf -- "$TRITON_PYTHON_BUILD_DIR"
  if ((DRY_RUN == 0)); then
    rm -rf -- "$TRITON_PYTHON_BUILD_DIR"
  fi
  run_logged triton-install "$TRITON_REPO_DIR" \
    python3 -m pip install --no-build-isolation -e python
  # shellcheck disable=SC1090
  source "$ENV_SCRIPT"
else
  echo "[triton] clean/install skipped"
fi

run_logged import-check "$TRITON_REPO_DIR" python3 -c \
  'import torch, triton; print("triton:", triton.__file__); print("torch:", torch.__version__)'

if ((DRY_RUN)); then
  echo "Dry-run complete; no pull, build, deletion, or install was performed."
else
  echo "Rebuild complete: $RUN_DIR"
fi
