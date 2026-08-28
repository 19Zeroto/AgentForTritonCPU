#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTFORTRITONCPU_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
AGENT_DIR="${AGENT_DIR:-$(cd "$AGENTFORTRITONCPU_DIR/.." && pwd)}"
TRITON_REPO_DIR="${TRITON_REPO_DIR:-$AGENT_DIR/triton-cpu}"
ENV_SCRIPT="${ENV_SCRIPT:-$AGENTFORTRITONCPU_DIR/skills/environment/scripts/triton-cpu-env.sh}"
DEFAULT_TESTS_DIR="$TRITON_REPO_DIR/FlagGems/tests"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$AGENT_DIR/logs/AgentForTritonCPU/all_flaggems_$RUN_ID}"
LOG_DIR="$RUN_DIR/pytest_logs"
ENV_LOG="$RUN_DIR/env.log"
MAX_JOBS=3
PYTEST_WORKERS=16
CACHE_ROOT="${CACHE_ROOT:-}"

# Entries may be paths relative to TESTS_DIR or plain basenames.
# Examples:
#   "test_special_ops.py"
#   "test_DSA/test_bin_topk.py"
EXCLUDED_TEST_FILES=(
)

usage() {
  cat <<EOF
Usage:
  ./run_all_flaggems_tests.sh [--cache-root PATH]

Environment overrides:
  TESTS_DIR    Test directory (default: $DEFAULT_TESTS_DIR)
  RUN_DIR      Run directory (default: \$AGENT_DIR/logs/AgentForTritonCPU/all_flaggems_<timestamp>)
  CACHE_ROOT   Cache root, same as --cache-root when set
EOF
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-root)
      [[ $# -ge 2 ]] || die "--cache-root requires a value"
      CACHE_ROOT="$2"
      shift 2
      ;;
    --cache-root=*)
      CACHE_ROOT="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

mkdir -p "$RUN_DIR" "$LOG_DIR" || die "cannot create run directory: $RUN_DIR"

# shellcheck disable=SC1091
if ! source "$ENV_SCRIPT" >"$ENV_LOG" 2>&1; then
  log "Environment setup log: $ENV_LOG"
  die "failed to source $ENV_SCRIPT"
fi

TESTS_DIR="${TESTS_DIR:-$DEFAULT_TESTS_DIR}"
TESTS_DIR="$(cd "$TESTS_DIR" 2>/dev/null && pwd)" || die "test directory not found: $TESTS_DIR"

if [[ -z "$CACHE_ROOT" ]]; then
  CACHE_ROOT="$RUN_DIR/triton_cache"
fi
mkdir -p "$CACHE_ROOT" || die "cannot create cache root: $CACHE_ROOT"
CACHE_ROOT="$(cd "$CACHE_ROOT" && pwd)" || die "cannot resolve cache root: $CACHE_ROOT"

mapfile -t TEST_FILES < <(
  cd "$TESTS_DIR" || exit 1
  find . -name 'test_*.py' -type f | sort | sed 's#^\./##'
)

is_excluded_file() {
  local rel_file="$1"
  local base_file
  local excluded

  base_file="$(basename "$rel_file")"
  for excluded in "${EXCLUDED_TEST_FILES[@]}"; do
    [[ -n "$excluded" ]] || continue
    if [[ "$rel_file" == "$excluded" || "$base_file" == "$excluded" ]]; then
      return 0
    fi
  done

  return 1
}

ALL_TEST_FILES=("${TEST_FILES[@]}")
TEST_FILES=()
SKIPPED_FILES=()
for rel_file in "${ALL_TEST_FILES[@]}"; do
  if is_excluded_file "$rel_file"; then
    SKIPPED_FILES+=("$rel_file")
  else
    TEST_FILES+=("$rel_file")
  fi
done

if [[ ${#ALL_TEST_FILES[@]} -eq 0 ]]; then
  die "no test_*.py files found under $TESTS_DIR"
fi
if [[ ${#TEST_FILES[@]} -eq 0 ]]; then
  die "all ${#ALL_TEST_FILES[@]} discovered test file(s) are excluded"
fi

log "Run dir: $RUN_DIR"
log "Test dir: $TESTS_DIR"
log "Cache root: $CACHE_ROOT"
log "Environment setup log: $ENV_LOG"
log "Pytest logs: $LOG_DIR"
log "Files: ${#TEST_FILES[@]} selected, ${#SKIPPED_FILES[@]} excluded, max concurrent pytest commands: $MAX_JOBS"
for rel_file in "${SKIPPED_FILES[@]}"; do
  log "SKIP  excluded $rel_file"
done

run_one_file() {
  local rel_file="$1"
  local cache_dir="$CACHE_ROOT/${rel_file%.py}"
  local log_file="$LOG_DIR/$rel_file.log"

  mkdir -p "$(dirname "$cache_dir")" "$cache_dir" "$(dirname "$log_file")" || return 2

  (
    printf '[%s] START %s\n' "$(timestamp)" "$rel_file"
    printf 'cwd: %s\n' "$TESTS_DIR"
    printf 'TRITON_CACHE_DIR: %s\n' "$cache_dir"
    printf 'command: TRITON_CACHE_DIR=%q OMP_NUM_THREAD=8 OMP_NUM_THREADS=8 pytest -n%s -q --tb=no %q\n' \
      "$cache_dir" "$PYTEST_WORKERS" "$rel_file"
    printf '\n'

    cd "$TESTS_DIR" || exit 2
    export TRITON_CACHE_DIR="$cache_dir"
    export OMP_NUM_THREAD=8
    export OMP_NUM_THREADS=8

    pytest -n"$PYTEST_WORKERS" -q --tb=no "$rel_file"
    status=$?

    printf '\n[%s] EXIT %s status=%s\n' "$(timestamp)" "$rel_file" "$status"
    exit "$status"
  ) >"$log_file" 2>&1
}

declare -A PID_TO_FILE=()
declare -A PID_TO_LOG=()

next_index=0
running=0
completed=0
failed=0
final_status=0
total=${#TEST_FILES[@]}

cleanup_children() {
  local pids=("${!PID_TO_FILE[@]}")
  if [[ ${#pids[@]} -gt 0 ]]; then
    log "Stopping ${#pids[@]} running pytest command(s)"
    kill "${pids[@]}" 2>/dev/null || true
  fi
}

trap 'cleanup_children; exit 130' INT TERM

start_next() {
  local rel_file="${TEST_FILES[$next_index]}"
  local log_file="$LOG_DIR/$rel_file.log"
  local pid

  run_one_file "$rel_file" &
  pid=$!
  PID_TO_FILE["$pid"]="$rel_file"
  PID_TO_LOG["$pid"]="$log_file"
  next_index=$((next_index + 1))
  running=$((running + 1))

  log "START [$next_index/$total] pid=$pid $rel_file"
}

wait_for_one() {
  local finished_pid=""
  local status=0
  local rel_file
  local log_file

  if wait -n -p finished_pid; then
    status=0
  else
    status=$?
  fi

  rel_file="${PID_TO_FILE[$finished_pid]:-<unknown>}"
  log_file="${PID_TO_LOG[$finished_pid]:-<unknown>}"
  unset 'PID_TO_FILE[$finished_pid]'
  unset 'PID_TO_LOG[$finished_pid]'

  running=$((running - 1))
  completed=$((completed + 1))

  if [[ "$status" -ne 0 ]]; then
    failed=$((failed + 1))
    final_status=1
    log "FAIL  [$completed/$total] status=$status pid=$finished_pid $rel_file log=$log_file"
  else
    log "PASS  [$completed/$total] status=0 pid=$finished_pid $rel_file log=$log_file"
  fi
}

while [[ "$next_index" -lt "$total" && "$running" -lt "$MAX_JOBS" ]]; do
  start_next
done

while [[ "$running" -gt 0 ]]; do
  wait_for_one
  while [[ "$next_index" -lt "$total" && "$running" -lt "$MAX_JOBS" ]]; do
    start_next
  done
done

trap - INT TERM

log "Finished: total=$total passed=$((total - failed)) failed=$failed"
log "Run dir: $RUN_DIR"
exit "$final_status"
