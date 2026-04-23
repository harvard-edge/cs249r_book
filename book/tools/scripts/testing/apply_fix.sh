#!/usr/bin/env bash
# apply_fix.sh
# Single-fix wrapper for the MLSysBook camera-ready overnight sweep.
#
# Usage:
#   apply_fix.sh --chapter <chapter-name> --vol <vol1|vol2> --format <html|pdf> \
#                --issue-code <code> --summary <one-line> \
#                --diagnosis <multi-line-string-or-@file> \
#                --patch-file <abs-path-to-patch>
#
# Required environment variables (must be set by orchestrator):
#   RUN_TS, WORKTREE, RUN_DIR
#
# Exit codes:
#   0  success — patch applied and committed
#   2  rejected — patch moved to ${RUN_DIR}/fixes/suggested/<id>.patch
#   3  invalid arguments
#   4  prerequisite failure (e.g. binder check missing)
#  99  forbidden remote-op pattern detected in this script

set -uo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SELF_DIR}/camera_ready_sweep.sh"
assert_no_remote_ops "$0"

# ---------------------- argument parsing -------------------------------------
CHAPTER=""
VOL=""
FMT=""
ISSUE_CODE=""
SUMMARY=""
DIAGNOSIS=""
PATCH_FILE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --chapter)     CHAPTER="$2"; shift 2 ;;
    --vol)         VOL="$2"; shift 2 ;;
    --format)      FMT="$2"; shift 2 ;;
    --issue-code)  ISSUE_CODE="$2"; shift 2 ;;
    --summary)     SUMMARY="$2"; shift 2 ;;
    --diagnosis)   DIAGNOSIS="$2"; shift 2 ;;
    --patch-file)  PATCH_FILE="$2"; shift 2 ;;
    *) echo "apply_fix.sh: unknown arg $1" >&2; exit 3 ;;
  esac
done

for v in CHAPTER VOL FMT ISSUE_CODE SUMMARY PATCH_FILE; do
  if [ -z "${!v}" ]; then
    echo "apply_fix.sh: missing required --${v,,}" >&2
    exit 3
  fi
done
case "${VOL}" in vol1|vol2) ;; *) echo "bad --vol ${VOL}" >&2; exit 3 ;; esac
case "${FMT}" in html|pdf)   ;; *) echo "bad --format ${FMT}" >&2; exit 3 ;; esac
[ -f "${PATCH_FILE}" ] || { echo "missing patch file ${PATCH_FILE}" >&2; exit 3; }
for v in RUN_TS WORKTREE RUN_DIR; do
  if [ -z "${!v:-}" ]; then echo "env var ${v} required" >&2; exit 3; fi
done

# Resolve diagnosis: support @file shorthand
if [[ "${DIAGNOSIS}" == @* ]]; then
  DIAGNOSIS_FILE="${DIAGNOSIS#@}"
  [ -f "${DIAGNOSIS_FILE}" ] || { echo "missing diagnosis file ${DIAGNOSIS_FILE}" >&2; exit 3; }
  DIAGNOSIS="$(cat "${DIAGNOSIS_FILE}")"
fi

cd "${WORKTREE}"

PATCH_ID="$(basename "${PATCH_FILE}" .patch)-$(date +%s)"
SUGGESTED_DEST="${RUN_DIR}/fixes/suggested/${PATCH_ID}.patch"
APPLIED_DIR="${RUN_DIR}/fixes/applied"
PER_FIX_LOG="${RUN_DIR}/logs/per-fix/${PATCH_ID}.log"

reject_to_suggested() {
  local reason="$1"
  mkdir -p "$(dirname "${SUGGESTED_DEST}")"
  cp "${PATCH_FILE}" "${SUGGESTED_DEST}"
  log_health "{\"event\":\"fix_reverted\",\"reason\":\"${reason//\"/\\\"}\",\"chapter\":\"${CHAPTER}\",\"issue_code\":\"${ISSUE_CODE}\",\"patch_id\":\"${PATCH_ID}\"}"
  echo "REJECTED: ${reason}" | tee -a "${PER_FIX_LOG}"
  exit 2
}

mkdir -p "${RUN_DIR}/logs/per-fix" "${APPLIED_DIR}" "$(dirname "${SUGGESTED_DEST}")"
echo "=== apply_fix.sh ${PATCH_ID} ===" > "${PER_FIX_LOG}"
echo "chapter=${CHAPTER} vol=${VOL} fmt=${FMT} issue=${ISSUE_CODE}" >> "${PER_FIX_LOG}"

# 2) git apply --check (dry run)
if ! git apply --check "${PATCH_FILE}" >>"${PER_FIX_LOG}" 2>&1; then
  reject_to_suggested "git_apply_check_failed"
fi

# 3) Forbidden-paths gate
FORBIDDEN_REGEXES=(
  '^_quarto.*\.yml$'
  '^_metadata\.yml$'
  '^book/quarto/config/'
  '\.bib$'
  '^book/cli/'
  '^book/tools/'
  '^book/vscode-ext/'
  '^\.cursor/'
  '^\.github/'
  '^Makefile$'
  '\.png$'
  '\.jpg$'
  '\.jpeg$'
  '\.svg$'
  '\.pdf$'
  '\.gif$'
  '\.webp$'
  '^\.claude/rules/'
)

CHANGED_FILES="$(git apply --numstat "${PATCH_FILE}" 2>>"${PER_FIX_LOG}" | awk '{print $3}')"
for cf in ${CHANGED_FILES}; do
  for rx in "${FORBIDDEN_REGEXES[@]}"; do
    if printf '%s' "${cf}" | grep -E -q "${rx}"; then
      reject_to_suggested "forbidden_path:${cf}:matches:${rx}"
    fi
  done
done

# 4) Size cap: total touched lines <= 40, file count <= 3
SIZE_TOTAL="$(git apply --numstat "${PATCH_FILE}" | awk '{adds+=$1; dels+=$2}END{print adds+dels}')"
FILE_COUNT="$(printf '%s\n' "${CHANGED_FILES}" | sed '/^$/d' | wc -l | tr -d ' ')"
if [ "${SIZE_TOTAL}" -gt 40 ]; then
  reject_to_suggested "patch_too_large:${SIZE_TOTAL}_lines"
fi
if [ "${FILE_COUNT}" -gt 3 ]; then
  reject_to_suggested "too_many_files:${FILE_COUNT}"
fi

# 5) Chapter cap: at most 5 fix(<chapter>) commits on this branch
EXISTING_COMMITS="$(git log --oneline "camera-ready/${RUN_TS}" 2>/dev/null \
  | awk -v c="fix(${CHAPTER})" 'index($2,c)==1' | wc -l | tr -d ' ' || echo 0)"
if [ "${EXISTING_COMMITS}" -ge 5 ]; then
  reject_to_suggested "chapter_cap_reached:${EXISTING_COMMITS}"
fi

# Resolve chapter qmd path (best effort)
CHAPTER_QMD="$(find "${WORKTREE}/book/quarto/contents/${VOL}" -type f -name "${CHAPTER}.qmd" | head -1)"
if [ -z "${CHAPTER_QMD}" ]; then
  CHAPTER_QMD="$(find "${WORKTREE}/book/quarto/contents/${VOL}" -type f -name '*.qmd' -path "*${CHAPTER}*" | head -1)"
fi

# Snapshot pre-apply violation counts (for delta detection)
snapshot_violations() {
  local out="$1"
  shift
  ( cd "${WORKTREE}/book" && ./binder check "$@" --json 2>/dev/null | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('total_issues',0))" ) > "${out}" 2>/dev/null \
    || echo 0 > "${out}"
}

PRE_REND="${PER_FIX_LOG}.pre-rend"
PRE_REFS="${PER_FIX_LOG}.pre-refs"
PRE_MITPRESS="${PER_FIX_LOG}.pre-mitpress"
# Post-refactor: the old `rendering` umbrella group was dissolved into semantic
# groups (markup, prose, punctuation, numbers, math, structure, code, tables,
# index). `check all` is the broadest equivalent and catches everything the old
# `rendering` call did plus more.
[ -n "${CHAPTER_QMD}" ] && snapshot_violations "${PRE_REND}" all --path "${CHAPTER_QMD}"
if grep -E -q '^\+.*@[a-zA-Z]' "${PATCH_FILE}" || grep -E -q '^\+.*\\cite' "${PATCH_FILE}"; then
  TOUCHED_REFS=1
  [ -n "${CHAPTER_QMD}" ] && snapshot_violations "${PRE_REFS}" refs --path "${CHAPTER_QMD}"
else
  TOUCHED_REFS=0
fi
[ -n "${CHAPTER_QMD}" ] && snapshot_violations "${PRE_MITPRESS}" punctuation --scope vs-period --path "${CHAPTER_QMD}"

# 6) Apply patch
if ! git apply "${PATCH_FILE}" >>"${PER_FIX_LOG}" 2>&1; then
  reject_to_suggested "git_apply_failed_after_check"
fi

revert_patch() {
  git apply -R "${PATCH_FILE}" >>"${PER_FIX_LOG}" 2>&1 || true
}

# 7) Post-apply rendering & refs checks vs snapshots
post_check_or_revert() {
  local label="$1" pre="$2"; shift 2
  if [ -z "${CHAPTER_QMD}" ] || [ ! -f "${pre}" ]; then return 0; fi
  local POST="${PER_FIX_LOG}.post-${label}"
  snapshot_violations "${POST}" "$@"
  local pre_n post_n
  pre_n="$(cat "${pre}")"
  post_n="$(cat "${POST}")"
  if [ "${post_n}" -gt "${pre_n}" ]; then
    revert_patch
    reject_to_suggested "new_${label}_violations:${pre_n}->${post_n}"
  fi
}
post_check_or_revert all "${PRE_REND}" all --path "${CHAPTER_QMD}"
[ "${TOUCHED_REFS}" -eq 1 ] && post_check_or_revert refs "${PRE_REFS}" refs --path "${CHAPTER_QMD}"

# 8) Style-specific scope checks (post-refactor: scopes live in semantic groups,
# no longer under `rendering --scope mitpress-*`).
#   group:scope pairs — one per style rule that was previously mitpress-*
for pair in \
    "punctuation:vs-period" \
    "numbers:percent-in-captions" \
    "punctuation:emdash" \
    "punctuation:eg-ie-comma" \
    "refs:capitalized" \
    "prose:above-below" \
    "punctuation:hyphen-range" \
    "prose:acknowledgements"; do
  group="${pair%%:*}"
  scope="${pair#*:}"
  PRE="${PER_FIX_LOG}.pre-${group}-${scope}"
  POST="${PER_FIX_LOG}.post-${group}-${scope}"
  [ -n "${CHAPTER_QMD}" ] || continue
  snapshot_violations "${PRE}" "${group}" --scope "${scope}" --path "${CHAPTER_QMD}" 2>/dev/null
  snapshot_violations "${POST}" "${group}" --scope "${scope}" --path "${CHAPTER_QMD}" 2>/dev/null
  pre_n="$(cat "${PRE}" 2>/dev/null || echo 0)"
  post_n="$(cat "${POST}" 2>/dev/null || echo 0)"
  if [ "${post_n}" -gt "${pre_n}" ]; then
    revert_patch
    reject_to_suggested "new_${group}_${scope}_violations:${pre_n}->${post_n}"
  fi
done

# 9) Rebuild the chapter in target format
case "${VOL}" in
  vol1) VOL_FLAG="--vol1" ;;
  vol2) VOL_FLAG="--vol2" ;;
esac
( cd "${WORKTREE}/book" && ./binder build "${FMT}" "${CHAPTER}" "${VOL_FLAG}" >>"${PER_FIX_LOG}" 2>&1 )
BUILD_RC=$?
if [ "${BUILD_RC}" -ne 0 ]; then
  revert_patch
  reject_to_suggested "build_failed:${FMT}:rc=${BUILD_RC}"
fi

# 10) Post-edit Gemini math re-validation
if grep -E -q '^\+.*(\$\$|\$[^$]+\$)' "${PATCH_FILE}"; then
  AFFECTED_LINES="$(grep -nE '\$\$|\$[^$]+\$' "${CHAPTER_QMD}" | awk -F: '{print $1}' | tr '\n' ',' | sed 's/,$//')"
  if [ -n "${AFFECTED_LINES}" ]; then
    if ! python3 "${SELF_DIR}/gemini_math_check.py" revalidate \
          --chapter-qmd "${CHAPTER_QMD}" \
          --equation-line-numbers "${AFFECTED_LINES}" >>"${PER_FIX_LOG}" 2>&1; then
      revert_patch
      reject_to_suggested "gemini_math_revalidation_failed"
    fi
  fi
fi

# 11) Commit
COMMIT_MSG_FILE="$(mktemp)"
{
  printf "fix(%s): %s — %s\n\n" "${CHAPTER}" "${ISSUE_CODE}" "${SUMMARY}"
  printf "Volume: %s\nFormat: %s\nIssue-Code: %s\nRun: %s\nPatch-Id: %s\n\n" \
    "${VOL}" "${FMT}" "${ISSUE_CODE}" "${RUN_TS}" "${PATCH_ID}"
  printf "Diagnosis:\n%s\n\n" "${DIAGNOSIS}"
  printf "Authored by: apply_fix.sh (camera-ready overnight sweep)\nLocal-only commit; never pushed.\n"
} > "${COMMIT_MSG_FILE}"

git add -A >>"${PER_FIX_LOG}" 2>&1
if ! git commit -F "${COMMIT_MSG_FILE}" >>"${PER_FIX_LOG}" 2>&1; then
  revert_patch
  rm -f "${COMMIT_MSG_FILE}"
  reject_to_suggested "git_commit_failed"
fi
rm -f "${COMMIT_MSG_FILE}"

NEW_SHA="$(git rev-parse HEAD)"

# 12) Save the committed patch
git format-patch -1 HEAD --stdout > "${APPLIED_DIR}/${NEW_SHA}.patch" 2>>"${PER_FIX_LOG}"

# 13) Append health entry
log_health "{\"event\":\"fix_applied\",\"sha\":\"${NEW_SHA}\",\"chapter\":\"${CHAPTER}\",\"vol\":\"${VOL}\",\"format\":\"${FMT}\",\"issue_code\":\"${ISSUE_CODE}\",\"patch_id\":\"${PATCH_ID}\"}"

# 14) Print the SHA on stdout
echo "${NEW_SHA}"
exit 0
