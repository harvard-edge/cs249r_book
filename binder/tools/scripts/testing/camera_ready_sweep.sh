#!/usr/bin/env bash
# camera_ready_sweep.sh
# Shared helpers for the MLSysBook camera-ready overnight build sweep.
# Source this file from other camera-ready scripts:
#   source "$(dirname "$0")/camera_ready_sweep.sh"
#
# LOCAL-ONLY. No remote git or gh operations are permitted by any caller.
#
# Required environment variables (set by the orchestrator):
#   RUN_TS    - run timestamp, e.g. 20260418-204343
#   WORKTREE  - absolute path to the camera-ready worktree
#   RUN_DIR   - absolute path to the run output directory

set -u

# -----------------------------------------------------------------------------
# now_iso : print current time as ISO 8601 UTC with timezone marker.
# -----------------------------------------------------------------------------
now_iso() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

# -----------------------------------------------------------------------------
# log_health <json_object_without_ts>
# Append a JSON line to ${RUN_DIR}/health.jsonl, injecting "ts" if missing.
# Caller passes a single-line JSON object string; we rewrite it via python3
# to ensure validity and to add ts.
# -----------------------------------------------------------------------------
log_health() {
  local payload="$1"
  if [ -z "${RUN_DIR:-}" ]; then
    echo "log_health: RUN_DIR is unset" >&2
    return 1
  fi
  local ts
  ts="$(now_iso)"
  python3 - "$payload" "$ts" "${RUN_DIR}/health.jsonl" <<'PY'
import json, sys
payload, ts, path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    obj = json.loads(payload)
except Exception as e:
    obj = {"event": "log_health_parse_error", "raw": payload, "error": str(e)}
if "ts" not in obj:
    obj["ts"] = ts
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(obj, separators=(",", ":")) + "\n")
PY
}

# -----------------------------------------------------------------------------
# assert_no_remote_ops [<file>]
# Refuse to continue if the calling script (or the explicitly-passed file)
# contains any forbidden remote/push/fetch patterns.
# When sourced, BASH_SOURCE[1] is the caller's script path.
# Exits 99 on violation.
# -----------------------------------------------------------------------------
assert_no_remote_ops() {
  local target="${1:-}"
  if [ -z "${target}" ]; then
    target="${BASH_SOURCE[1]:-${0}}"
  fi
  if [ ! -f "${target}" ]; then
    echo "assert_no_remote_ops: cannot find ${target}" >&2
    return 99
  fi
  # Patterns are intentionally conservative; matches must be exact tokens that
  # would actually mutate or contact a remote.
  local pattern
  pattern='git[[:space:]]+push|git[[:space:]]+pull|git[[:space:]]+fetch[[:space:]]|git[[:space:]]+remote[[:space:]]+(add|set-url|update|set-head)|git[[:space:]]+config[[:space:]]+remote\.|gh[[:space:]]+pr[[:space:]]+(create|push|merge)|gh[[:space:]]+repo[[:space:]]+(create|clone|sync|fork)'
  # Scan only the script body, not comments. Use grep -E with -v on commented
  # lines to reduce false positives. We deliberately allow the pattern itself
  # to appear in this helper file by scanning the *caller*, not self.
  if [ "${target}" = "${BASH_SOURCE[0]}" ]; then
    return 0
  fi
  if grep -E -n "${pattern}" "${target}" | grep -E -v '^[^:]+:[[:space:]]*[0-9]+:[[:space:]]*#' >/dev/null 2>&1; then
    echo "assert_no_remote_ops: forbidden remote-op pattern found in ${target}" >&2
    grep -E -n "${pattern}" "${target}" | grep -E -v '^[^:]+:[[:space:]]*[0-9]+:[[:space:]]*#' >&2
    exit 99
  fi
  return 0
}

# Self-check on source: block any caller that already has remote ops in it.
# Skip if BASH_SOURCE has only one frame (sourced interactively).
if [ "${#BASH_SOURCE[@]}" -gt 1 ]; then
  assert_no_remote_ops "${BASH_SOURCE[1]}"
fi
