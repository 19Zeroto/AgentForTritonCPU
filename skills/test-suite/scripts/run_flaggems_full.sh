#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTFORTRITONCPU_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
AGENT_DIR="${AGENT_DIR:-$(cd "$AGENTFORTRITONCPU_DIR/.." && pwd)}"
ENV_SCRIPT="${ENV_SCRIPT:-$AGENTFORTRITONCPU_DIR/skills/environment/scripts/triton-cpu-env.sh}"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$AGENT_DIR/logs/AgentForTritonCPU/flaggems_full_$RUN_ID}"

if [[ "${1:-}" == "--background" ]]; then
  shift
  mkdir -p "$RUN_DIR"
  export RUN_ID RUN_DIR
  nohup "$SCRIPT_PATH" "$@" > "$RUN_DIR/main.log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" > "$RUN_DIR/pid"
  {
    echo "pid: $pid"
    echo "run_id: $RUN_ID"
    echo "run_dir: $RUN_DIR"
    echo "main_log: $RUN_DIR/main.log"
    echo
    echo "launcher_command:"
    printf 'RUN_ID=%q RUN_DIR=%q nohup %q' "$RUN_ID" "$RUN_DIR" "$SCRIPT_PATH"
    printf ' %q' "$@"
    printf ' > %q 2>&1 &\n' "$RUN_DIR/main.log"
  } > "$RUN_DIR/launcher.txt"
  echo "Started FlagGems full test run"
  echo "PID: $pid"
  echo "Run dir: $RUN_DIR"
  echo "Main log: $RUN_DIR/main.log"
  exit 0
fi

cd "$SCRIPT_DIR"

mkdir -p "$RUN_DIR"

# shellcheck disable=SC1091
source "$ENV_SCRIPT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

TESTS_DIR="${TESTS_DIR:-$TRITON_REPO_DIR/FlagGems/tests}"
FILES="${FILES:-1}"
PYTEST_THREADS="${PYTEST_THREADS:-3}"
EXCLUDE_REGEX="${EXCLUDE_REGEX:-(^|/)test_(special|norm)_ops\\.py$}"
STATE_FILE="${STATE_FILE:-$RUN_DIR/state.json}"
PYTEST_LOG_DIR="${PYTEST_LOG_DIR:-$RUN_DIR/pytest_logs}"

mkdir -p "$PYTEST_LOG_DIR"

RUN_CMD=(
  python3 "$SCRIPT_DIR/run_simple_tests.py"
  -d "$TESTS_DIR"
  -N "$FILES"
  -M "$PYTEST_THREADS"
  -e "$EXCLUDE_REGEX"
  -s "$STATE_FILE"
  --log-dir "$PYTEST_LOG_DIR"
  --save-all-logs
)

if [[ "${RESUME:-0}" == "1" ]]; then
  RUN_CMD+=(-r)
fi

write_git_revision() {
  local label="$1"
  local dir="$2"
  if git -C "$dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf '%s: %s\n' "$label" "$(git -C "$dir" rev-parse HEAD)"
    printf '%s_branch: %s\n' "$label" "$(git -C "$dir" rev-parse --abbrev-ref HEAD)"
  else
    printf '%s: <not a git worktree>\n' "$label"
  fi
}

{
  echo "created_at: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "cwd: $SCRIPT_DIR"
  echo "run_id: $RUN_ID"
  echo "run_dir: $RUN_DIR"
  echo "tests_dir: $TESTS_DIR"
  echo "state_file: $STATE_FILE"
  echo "pytest_log_dir: $PYTEST_LOG_DIR"
  echo "exclude_regex: $EXCLUDE_REGEX"
  echo "resume: ${RESUME:-0}"
  echo
  echo "command:"
  printf '%q ' "${RUN_CMD[@]}"
  echo
  echo
  echo "environment:"
  env | sort | grep -E '^(AGENT_DIR|VENV_DIR|LLVM_INSTALL_DIR|TRITON_REPO_DIR|TRITON_SHARED_[A-Z0-9_]*|TRITON_CACHE_DIR|GEMS_VENDOR|OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_NUM_THREADS|PYTHONPATH|PATH)=' || true
  echo
  echo "git_revisions:"
  write_git_revision "AgentForTritonCPU" "$AGENTFORTRITONCPU_DIR"
  write_git_revision "triton_cpu" "$TRITON_REPO_DIR"
  if [[ -d "$TRITON_REPO_DIR/FlagGems" ]]; then
    write_git_revision "flaggems" "$TRITON_REPO_DIR/FlagGems"
  fi
} > "$RUN_DIR/command.txt"

included_file_list="$RUN_DIR/included_files.txt"
skipped_file_list="$RUN_DIR/skipped_files.txt"
: > "$included_file_list"
: > "$skipped_file_list"
while IFS= read -r test_file; do
  rel="${test_file#"$TESTS_DIR"/}"
  if [[ "$rel" =~ $EXCLUDE_REGEX ]]; then
    printf '%s\n' "$rel" >> "$skipped_file_list"
  else
    printf '%s\n' "$rel" >> "$included_file_list"
  fi
done < <(find "$TESTS_DIR" -name 'test_*.py' -type f | sort)

echo "Command metadata: $RUN_DIR/command.txt"
echo "Included files: $included_file_list"
echo "Skipped files: $skipped_file_list"
echo "State file: $STATE_FILE"
echo "Per-file logs: $PYTEST_LOG_DIR"
echo
echo "Running command:"
printf '%q ' "${RUN_CMD[@]}"
echo

set +e
"${RUN_CMD[@]}"
status=$?
set -e

printf '%s\n' "$status" > "$RUN_DIR/exit_code"
echo "Finished with exit code: $status"
exit "$status"
