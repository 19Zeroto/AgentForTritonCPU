#!/usr/bin/env bash

# PURPOSE
#   Locate the first Triton-CPU commit that makes one pytest target fail.
#
# CONFIGURE
#   1. Set DEFAULT_TEST_TARGET to a test file or full pytest nodeid.
#   2. Set DEFAULT_GOOD_COMMIT to a known-good commit.
#   3. Keep DEFAULT_BAD_COMMIT as HEAD, or set another known-bad commit.
#   Command-line options override all defaults; run with --help for details.
#
# RESULT RULES
#   pytest exit 0   -> good commit
#   pytest nonzero  -> bad commit
#   build, install, collection, or environment failure -> skipped commit (125)
#
# SIDE EFFECTS
#   Checks out multiple commits in DEFAULT_REPO, rebuilds editable Triton and
#   FlagGems installs, updates the SLEEF submodule, and writes per-commit logs.
#   Tracked changes in DEFAULT_REPO are rejected. Original checkout is restored.
#
# QUICK START
#   bash scripts/bisect_triton_commit.sh \
#     --good <good_commit> \
#     --test 'FlagGems/tests/test_ops.py::test_name'

set -uo pipefail

# User-facing defaults. Keep these near the top for easy manual editing.
DEFAULT_TEST_TARGET=""
DEFAULT_GOOD_COMMIT=""
DEFAULT_BAD_COMMIT="HEAD"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
AGENTFORTRITONCPU_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
AGENT_DIR="${AGENT_DIR:-$(cd "$AGENTFORTRITONCPU_DIR/.." && pwd)}"
DEFAULT_REPO="${TRITON_REPO_DIR:-$AGENT_DIR/triton-cpu}"
DEFAULT_ENV_SCRIPT="$AGENTFORTRITONCPU_DIR/skills/environment/scripts/triton-cpu-env.sh"

MODE="run"
REPO="$DEFAULT_REPO"
ENV_SCRIPT="$DEFAULT_ENV_SCRIPT"
BUILD_ENV_SCRIPT=""
GOOD_COMMIT="$DEFAULT_GOOD_COMMIT"
BAD_COMMIT="$DEFAULT_BAD_COMMIT"
TEST_TARGET="$DEFAULT_TEST_TARGET"
LOG_ROOT=""
REPEAT=1
XDIST=0
NO_BUILD=0
NO_VERIFY_ENDPOINTS=0
FIRST_PARENT=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/bisect_triton_commit.sh --good <good_commit> --test <pytest_target> [options]

Find the first Triton-CPU commit that breaks a pytest target.

Required:
  --good <commit>             Known-good starting commit.
  --test <pytest-target>      Test file or nodeid relative to the repo root.
                              Example: FlagGems/tests/test_ops.py::test_name

Options:
  --bad <commit>              Known-bad commit (default: HEAD).
  --repo <path>               Triton-CPU repo (default: $AGENT_DIR/triton-cpu).
  --env-script <path>         Environment script to source before build/test
                              (default: skills/environment/scripts/triton-cpu-env.sh).
  --build-env-script <path>   Extra script sourced after --env-script before each build.
                              Use this to override LLVM/Triton build variables.
  --repeat <N>                Run pytest N times per commit; any failure is bad
                              (default: 1).
  --xdist <N>                 Add pytest -n N when N > 0 (default: 0).
  --log-dir <path>            Log root (default: $AGENT_DIR/logs/AgentForTritonCPU/).
  --first-parent              Ask git bisect to follow first-parent history.
  --no-build                  Skip Triton rebuild before pytest.
  --no-verify-endpoints       Do not pre-test the good and bad endpoints.
  -h, --help                  Show this help.

Environment:
  Exported variables from the caller are inherited by the rebuild and pytest.
  The default env script uses an existing LLVM_INSTALL_DIR if one is already set.
  If you need final overrides before compiling Triton, put them in a shell script
  and pass it with --build-env-script.

Example:
  export LLVM_INSTALL_DIR="$HOME/agent/llvm-project/install"
  bash scripts/bisect_triton_commit.sh \
    --good <good_commit> \
    --test 'FlagGems/tests/test_reduction_ops.py::test_cross_entropy_loss'

Internal:
  --probe                     Judge the currently checked-out commit for git bisect run.
EOF
}

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 2
}

parse_positive_int() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ || "$value" -lt 1 ]]; then
    die "$name must be a positive integer: $value"
  fi
}

parse_nonnegative_int() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    die "$name must be a non-negative integer: $value"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --probe)
      MODE="probe"
      shift
      ;;
    --good)
      [[ $# -ge 2 ]] || die "--good requires a value"
      GOOD_COMMIT="$2"
      shift 2
      ;;
    --bad)
      [[ $# -ge 2 ]] || die "--bad requires a value"
      BAD_COMMIT="$2"
      shift 2
      ;;
    --test)
      [[ $# -ge 2 ]] || die "--test requires a value"
      TEST_TARGET="$2"
      shift 2
      ;;
    --repo)
      [[ $# -ge 2 ]] || die "--repo requires a value"
      REPO="$2"
      shift 2
      ;;
    --env-script)
      [[ $# -ge 2 ]] || die "--env-script requires a value"
      ENV_SCRIPT="$2"
      shift 2
      ;;
    --build-env-script)
      [[ $# -ge 2 ]] || die "--build-env-script requires a value"
      BUILD_ENV_SCRIPT="$2"
      shift 2
      ;;
    --repeat)
      [[ $# -ge 2 ]] || die "--repeat requires a value"
      REPEAT="$2"
      parse_positive_int "--repeat" "$REPEAT"
      shift 2
      ;;
    --xdist)
      [[ $# -ge 2 ]] || die "--xdist requires a value"
      XDIST="$2"
      parse_nonnegative_int "--xdist" "$XDIST"
      shift 2
      ;;
    --log-dir)
      [[ $# -ge 2 ]] || die "--log-dir requires a value"
      LOG_ROOT="$2"
      shift 2
      ;;
    --first-parent)
      FIRST_PARENT=1
      shift
      ;;
    --no-build)
      NO_BUILD=1
      shift
      ;;
    --no-verify-endpoints)
      NO_VERIFY_ENDPOINTS=1
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

REPO="$(cd "$REPO" 2>/dev/null && pwd)" || die "repo not found: $REPO"
if [[ -n "$ENV_SCRIPT" ]]; then
  ENV_SCRIPT="$(cd "$(dirname "$ENV_SCRIPT")" 2>/dev/null && pwd)/$(basename "$ENV_SCRIPT")" \
    || die "env script directory not found: $ENV_SCRIPT"
fi
if [[ -n "$BUILD_ENV_SCRIPT" ]]; then
  BUILD_ENV_SCRIPT="$(cd "$(dirname "$BUILD_ENV_SCRIPT")" 2>/dev/null && pwd)/$(basename "$BUILD_ENV_SCRIPT")" \
    || die "build env script directory not found: $BUILD_ENV_SCRIPT"
fi

if [[ -z "$LOG_ROOT" ]]; then
  LOG_ROOT="$AGENT_DIR/logs/AgentForTritonCPU/bisect-triton-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$LOG_ROOT" || die "cannot create log dir: $LOG_ROOT"
LOG_ROOT="$(cd "$LOG_ROOT" && pwd)"

run_git() {
  git -C "$REPO" "$@"
}

tail_log() {
  local file="$1"
  local lines="${2:-80}"
  if [[ -f "$file" ]]; then
    log "Last $lines lines of $file:"
    tail -n "$lines" "$file"
  fi
}

source_env_for_probe() {
  local env_log="$1"

  export TRITON_REPO_DIR="$REPO"

  if [[ -n "$ENV_SCRIPT" ]]; then
    if [[ ! -f "$ENV_SCRIPT" ]]; then
      log "PROBE skip: env script not found: $ENV_SCRIPT"
      return 125
    fi
    # shellcheck disable=SC1090
    if ! source "$ENV_SCRIPT" >>"$env_log" 2>&1; then
      log "PROBE skip: failed to source env script: $ENV_SCRIPT"
      tail_log "$env_log"
      return 125
    fi
  fi

  if [[ -n "$BUILD_ENV_SCRIPT" ]]; then
    if [[ ! -f "$BUILD_ENV_SCRIPT" ]]; then
      log "PROBE skip: build env script not found: $BUILD_ENV_SCRIPT"
      return 125
    fi
    # shellcheck disable=SC1090
    if ! source "$BUILD_ENV_SCRIPT" >>"$env_log" 2>&1; then
      log "PROBE skip: failed to source build env script: $BUILD_ENV_SCRIPT"
      tail_log "$env_log"
      return 125
    fi
  fi

  # Keep FlagGems imports tied to the commit currently checked out by bisect.
  # The Python environment may already have another FlagGems editable install.
  if [[ -d "$REPO/FlagGems/src" ]]; then
    export PYTHONPATH="$REPO/FlagGems/src${PYTHONPATH:+:$PYTHONPATH}"
  fi

  return 0
}

write_env_snapshot() {
  local snapshot="$1"
  {
    echo "repo=$REPO"
    echo "commit=$(run_git rev-parse HEAD)"
    echo "python=$(command -v python3 || true)"
    python3 --version 2>&1 || true
    echo
    env | grep -E '^(AGENT_DIR|VENV_DIR|CONDA_|LLVM_|MLIR_|TRITON_|CMAKE_PREFIX_PATH|PYTHONPATH|PATH|GEMS_VENDOR|MAX_JOBS)=' | sort || true
  } >"$snapshot"
}

refresh_triton_shared_opt_path() {
  local found
  found="$(find "$REPO/python/build" -path '*triton-shared-opt' -type f 2>/dev/null | head -n 1 || true)"
  if [[ -n "$found" ]]; then
    export TRITON_SHARED_OPT_PATH="$found"
  fi
}

probe_current_commit() {
  local full_sha short_sha probe_dir env_log build_log collect_log snapshot_log cache_dir
  full_sha="$(run_git rev-parse HEAD)" || return 125
  short_sha="$(run_git rev-parse --short=12 HEAD)" || return 125
  probe_dir="$LOG_ROOT/probes/$short_sha"
  env_log="$probe_dir/env.log"
  build_log="$probe_dir/build.log"
  collect_log="$probe_dir/collect.log"
  snapshot_log="$probe_dir/env-snapshot.log"
  cache_dir="$probe_dir/triton-cache"

  mkdir -p "$probe_dir" || return 125
  : >"$env_log"

  log "PROBE start commit=$full_sha"

  if ! source_env_for_probe "$env_log"; then
    return 125
  fi

  write_env_snapshot "$snapshot_log"

  if ! run_git submodule update --init third_party/sleef >"$probe_dir/submodule.log" 2>&1; then
    log "PROBE skip commit=$short_sha reason=submodule-update-failed"
    tail_log "$probe_dir/submodule.log"
    return 125
  fi

  if [[ "$NO_BUILD" -eq 0 ]]; then
    log "PROBE build commit=$short_sha"
    if ! (cd "$REPO" && python3 -m pip install --no-build-isolation -e python) >"$build_log" 2>&1; then
      log "PROBE skip commit=$short_sha reason=build-failed log=$build_log"
      tail_log "$build_log"
      return 125
    fi
    refresh_triton_shared_opt_path
  else
    log "PROBE build commit=$short_sha skipped (--no-build)"
    : >"$build_log"
  fi

  if [[ -d "$REPO/FlagGems" ]]; then
    log "PROBE install FlagGems commit=$short_sha"
    if ! (cd "$REPO" && python3 -m pip install --no-build-isolation --no-deps -e FlagGems) >"$probe_dir/flaggems-install.log" 2>&1; then
      log "PROBE skip commit=$short_sha reason=flaggems-install-failed log=$probe_dir/flaggems-install.log"
      tail_log "$probe_dir/flaggems-install.log"
      return 125
    fi
  fi

  rm -rf "$cache_dir"
  mkdir -p "$cache_dir" || return 125
  export TRITON_CACHE_DIR="$cache_dir"
  export PYTHONUNBUFFERED=1
  export PYTHONWARNINGS=ignore
  export COVERAGE_PROCESS_START=""

  local target_file target_path
  target_file="${TEST_TARGET%%::*}"
  target_path="$REPO/$TEST_TARGET"
  if [[ ! -f "$REPO/$target_file" ]]; then
    log "PROBE skip commit=$short_sha reason=test-file-not-found path=$REPO/$target_file"
    return 125
  fi

  local collect_cmd
  collect_cmd=(python3 -m pytest "$target_path" --collect-only -q -p no:cacheprovider)
  log "PROBE collect commit=$short_sha test=$TEST_TARGET"
  if ! "${collect_cmd[@]}" >"$collect_log" 2>&1; then
    log "PROBE skip commit=$short_sha reason=collect-failed log=$collect_log"
    tail_log "$collect_log"
    return 125
  fi

  local collected
  collected="$(grep -c '::' "$collect_log" || true)"
  if [[ "$collected" -eq 0 ]]; then
    log "PROBE skip commit=$short_sha reason=no-tests-collected log=$collect_log"
    tail_log "$collect_log"
    return 125
  fi
  log "PROBE collected commit=$short_sha tests=$collected"

  local run_idx rc run_log
  local pytest_cmd
  pytest_cmd=(python3 -m pytest "$target_path" --tb=short -q -p no:cacheprovider)
  if [[ "$XDIST" -gt 0 ]]; then
    pytest_cmd+=(-n "$XDIST")
  fi

  run_idx=1
  while [[ "$run_idx" -le "$REPEAT" ]]; do
    run_log="$probe_dir/pytest-$run_idx.log"
    log "PROBE pytest commit=$short_sha repeat=$run_idx/$REPEAT"
    "${pytest_cmd[@]}" >"$run_log" 2>&1
    rc=$?
    if [[ "$rc" -ne 0 ]]; then
      log "PROBE result commit=$short_sha result=bad repeat=$run_idx exit=$rc log=$run_log"
      tail_log "$run_log"
      return 1
    fi
    run_idx=$((run_idx + 1))
  done

  log "PROBE result commit=$short_sha result=good"
  return 0
}

restore_original_checkout() {
  local original_branch="$1"
  local original_head="$2"
  local bisect_started="$3"

  if [[ "$bisect_started" -eq 1 ]]; then
    run_git bisect reset >/dev/null 2>&1 || true
    return
  fi

  if [[ -n "$original_branch" ]]; then
    run_git checkout "$original_branch" >/dev/null 2>&1 || true
  elif [[ -n "$original_head" ]]; then
    run_git checkout --detach "$original_head" >/dev/null 2>&1 || true
  fi
}

verify_endpoint() {
  local label="$1"
  local commit="$2"
  local expect="$3"
  local rc

  log "VERIFY $label checkout $commit"
  if ! run_git checkout --detach "$commit" >/dev/null 2>&1; then
    die "failed to checkout $label commit: $commit"
  fi

  probe_current_commit
  rc=$?

  if [[ "$expect" == "good" ]]; then
    if [[ "$rc" -ne 0 ]]; then
      die "known-good endpoint did not pass: $commit (probe exit $rc)"
    fi
  else
    if [[ "$rc" -eq 0 ]]; then
      die "known-bad endpoint passed unexpectedly: $commit"
    fi
    if [[ "$rc" -eq 125 ]]; then
      die "known-bad endpoint is untestable: $commit"
    fi
  fi
}

ensure_clean_tracked_tree() {
  if ! run_git diff --quiet --; then
    die "target repo has unstaged tracked changes; commit/stash them before bisect"
  fi
  if ! run_git diff --cached --quiet --; then
    die "target repo has staged changes; commit/stash them before bisect"
  fi
}

ensure_no_active_bisect() {
  local bisect_log
  bisect_log="$(run_git rev-parse --git-path BISECT_LOG)"
  if [[ -f "$bisect_log" ]]; then
    die "target repo already has an active bisect; run 'git -C $REPO bisect reset' first"
  fi
}

run_bisect() {
  [[ -n "$GOOD_COMMIT" ]] || die "--good <good_commit> is required"
  [[ -n "$TEST_TARGET" ]] || die "--test <pytest_target> is required"
  [[ -d "$REPO/.git" ]] || die "not a git repository: $REPO"

  ensure_no_active_bisect
  ensure_clean_tracked_tree

  local good_sha bad_sha original_branch original_head bisect_started
  good_sha="$(run_git rev-parse --verify "$GOOD_COMMIT^{commit}")" \
    || die "cannot resolve good commit: $GOOD_COMMIT"
  bad_sha="$(run_git rev-parse --verify "$BAD_COMMIT^{commit}")" \
    || die "cannot resolve bad commit: $BAD_COMMIT"

  if ! run_git merge-base --is-ancestor "$good_sha" "$bad_sha"; then
    die "good commit is not an ancestor of bad commit: good=$good_sha bad=$bad_sha"
  fi

  original_branch="$(run_git symbolic-ref --quiet --short HEAD || true)"
  original_head="$(run_git rev-parse --verify HEAD)"
  bisect_started=0

  trap 'rc=$?; restore_original_checkout "$original_branch" "$original_head" "$bisect_started"; exit "$rc"' EXIT INT TERM

  {
    echo "repo=$REPO"
    echo "good=$good_sha"
    echo "bad=$bad_sha"
    echo "test_target=$TEST_TARGET"
    echo "repeat=$REPEAT"
    echo "xdist=$XDIST"
    echo "no_build=$NO_BUILD"
    echo "env_script=$ENV_SCRIPT"
    echo "build_env_script=${BUILD_ENV_SCRIPT:-<none>}"
    echo "first_parent=$FIRST_PARENT"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  } >"$LOG_ROOT/config.txt"

  log "Logs: $LOG_ROOT"
  log "Range: good=$good_sha bad=$bad_sha"

  if [[ "$NO_VERIFY_ENDPOINTS" -eq 0 ]]; then
    verify_endpoint "good" "$good_sha" "good"
    verify_endpoint "bad" "$bad_sha" "bad"

    if [[ -n "$original_branch" ]]; then
      run_git checkout "$original_branch" >/dev/null 2>&1 \
        || die "failed to restore original branch before bisect: $original_branch"
    else
      run_git checkout --detach "$original_head" >/dev/null 2>&1 \
        || die "failed to restore original detached HEAD before bisect: $original_head"
    fi
  else
    log "Endpoint verification skipped (--no-verify-endpoints)"
  fi

  local start_args=(bisect start)
  if [[ "$FIRST_PARENT" -eq 1 ]]; then
    start_args+=(--first-parent)
  fi
  start_args+=("$bad_sha" "$good_sha")

  log "Starting git bisect"
  if ! run_git "${start_args[@]}" >"$LOG_ROOT/bisect-start.log" 2>&1; then
    tail_log "$LOG_ROOT/bisect-start.log"
    die "git bisect start failed"
  fi
  bisect_started=1

  local probe_cmd=(
    "$SELF"
    --probe
    --repo "$REPO"
    --env-script "$ENV_SCRIPT"
    --repeat "$REPEAT"
    --xdist "$XDIST"
    --test "$TEST_TARGET"
    --log-dir "$LOG_ROOT"
  )
  if [[ -n "$BUILD_ENV_SCRIPT" ]]; then
    probe_cmd+=(--build-env-script "$BUILD_ENV_SCRIPT")
  fi
  if [[ "$NO_BUILD" -eq 1 ]]; then
    probe_cmd+=(--no-build)
  fi

  log "Running git bisect run"
  run_git bisect run "${probe_cmd[@]}" >"$LOG_ROOT/bisect-run.log" 2>&1
  local bisect_rc=$?
  cat "$LOG_ROOT/bisect-run.log"
  run_git bisect log >"$LOG_ROOT/git-bisect.log" 2>&1 || true

  if [[ "$bisect_rc" -ne 0 ]]; then
    log "git bisect run exited with $bisect_rc"
  fi

  local first_bad
  first_bad="$(awk '/is the first bad commit/ {print $1; exit}' "$LOG_ROOT/bisect-run.log" || true)"
  if [[ -z "$first_bad" ]]; then
    log "Could not determine a unique first bad commit. See: $LOG_ROOT/bisect-run.log"
    restore_original_checkout "$original_branch" "$original_head" "$bisect_started"
    bisect_started=0
    trap - EXIT INT TERM
    return 1
  fi

  log "FIRST_BAD_COMMIT=$first_bad"
  run_git show --stat --oneline --decorate "$first_bad" >"$LOG_ROOT/first-bad.txt" 2>&1 || true
  cat "$LOG_ROOT/first-bad.txt"

  restore_original_checkout "$original_branch" "$original_head" "$bisect_started"
  bisect_started=0
  trap - EXIT INT TERM
}

if [[ "$MODE" == "probe" ]]; then
  probe_current_commit
  exit $?
fi

run_bisect
