#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
LABS_ROOT="${LABS_ROOT:-labs}"
LABS_DIR="${REPO_ROOT}/${LABS_ROOT}"
MLSYSIM_DIR="${REPO_ROOT}/mlsysim"

if [ ! -d "${LABS_DIR}" ]; then
  echo "ERROR: Labs root does not exist: ${LABS_DIR}" >&2
  exit 1
fi

if [ ! -f "${LABS_DIR}/_quarto.yml" ]; then
  echo "ERROR: ${LABS_DIR}/_quarto.yml is required for the labs site build." >&2
  exit 1
fi

if [ ! -d "${MLSYSIM_DIR}" ]; then
  echo "ERROR: mlsysim package directory is missing: ${MLSYSIM_DIR}" >&2
  exit 1
fi

if ! command -v marimo >/dev/null 2>&1; then
  echo "ERROR: marimo CLI is required. Install labs/requirements.txt first." >&2
  exit 1
fi

if ! command -v quarto >/dev/null 2>&1; then
  echo "ERROR: quarto CLI is required for the labs site build." >&2
  exit 1
fi

python3 -m build --wheel "${MLSYSIM_DIR}"

rm -rf "${LABS_DIR}/_wasm_build" "${LABS_DIR}/_build"
mkdir -p "${LABS_DIR}/_wasm_build/wheels"
cp "${MLSYSIM_DIR}"/dist/mlsysim-*.whl "${LABS_DIR}/_wasm_build/wheels/"

expected=0
exported=0
failed=""

for vol in vol1 vol2; do
  mkdir -p "${LABS_DIR}/_wasm_build/${vol}"
  for lab in "${LABS_DIR}/${vol}"/lab_*.py; do
    name="$(basename "${lab}" .py)"
    expected=$((expected + 1))
    echo "Exporting ${vol}/${name}..."
    if marimo export html-wasm "${lab}" \
      -o "${LABS_DIR}/_wasm_build/${vol}/${name}/index.html" \
      --mode run \
      --no-show-code; then
      exported=$((exported + 1))
    else
      failed="${failed} ${vol}/${name}"
    fi
  done
done

echo "WASM export summary: ${exported}/${expected} notebooks exported"
if [ -n "${failed}" ]; then
  echo "ERROR: Failed exports:${failed}" >&2
  exit 1
fi

(cd "${LABS_DIR}" && quarto render && touch _build/.nojekyll)

cp -r "${LABS_DIR}/_wasm_build/vol1" "${LABS_DIR}/_build/vol1"
cp -r "${LABS_DIR}/_wasm_build/vol2" "${LABS_DIR}/_build/vol2"
cp -r "${LABS_DIR}/_wasm_build/wheels" "${LABS_DIR}/_build/wheels"
# Duplicate wheels into vol directories to satisfy Pyodide worker relative paths
cp -r "${LABS_DIR}/_wasm_build/wheels" "${LABS_DIR}/_build/vol1/wheels"
cp -r "${LABS_DIR}/_wasm_build/wheels" "${LABS_DIR}/_build/vol2/wheels"
cp "${LABS_DIR}/site.webmanifest" "${LABS_DIR}/_build/site.webmanifest"

wasm_count="$(find "${LABS_DIR}/_build/vol1" "${LABS_DIR}/_build/vol2" -name "index.html" | wc -l | tr -d ' ')"
echo "WASM notebooks in final build: ${wasm_count}"
if [ "${wasm_count}" -ne "${expected}" ]; then
  echo "ERROR: Expected ${expected} WASM notebooks in build output, got ${wasm_count}" >&2
  exit 1
fi

test -f "${LABS_DIR}/_build/index.html"
test -f "${LABS_DIR}/_build/site.webmanifest"
test -f "${LABS_DIR}/_build/assets/images/favicon.svg"
test -f "${LABS_DIR}/_build/assets/images/social-card.svg"
echo "Labs site build complete: ${LABS_DIR}/_build"
