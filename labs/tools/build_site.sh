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
python3 -m build --wheel "${LABS_DIR}" --outdir "${REPO_ROOT}/wheels"

# Verify the built wheel version matches what labs reference via micropip.
# A mismatch causes BadZipFile in the browser (micropip fetches a 404 HTML page).
BUILT_VERSION=$(python3 -c "
import pathlib
try:
    import tomllib
except ImportError:
    import tomli as tomllib
p = pathlib.Path('${MLSYSIM_DIR}/pyproject.toml')
print(tomllib.loads(p.read_text())['project']['version'])
")
BUILT_WHL="${MLSYSIM_DIR}/dist/mlsysim-${BUILT_VERSION}-py3-none-any.whl"
if [ ! -f "${BUILT_WHL}" ]; then
  echo "ERROR: Expected wheel not found after build: ${BUILT_WHL}" >&2
  exit 1
fi
mkdir -p "${REPO_ROOT}/wheels"
cp "${BUILT_WHL}" "${REPO_ROOT}/wheels/"

# Confirm every lab references the built wheel version.
BAD_LABS=""
for lab in "${LABS_DIR}"/vol*/lab_*.py; do
  if ! grep -q "mlsysim-${BUILT_VERSION}-py3-none-any.whl" "${lab}"; then
    BAD_LABS="${BAD_LABS} ${lab}"
  fi
done
if [ -n "${BAD_LABS}" ]; then
  echo "ERROR: Built wheel is ${BUILT_VERSION}, but these labs reference a different wheel:" >&2
  for lab in ${BAD_LABS}; do
    echo "  - ${lab}" >&2
  done
  echo "Update all micropip.install() calls to reference mlsysim-${BUILT_VERSION}-py3-none-any.whl" >&2
  exit 1
fi
echo "Wheel version check passed: ${BUILT_VERSION}"

LAB_HELPER_VERSION=$(python3 -c "
import pathlib
try:
    import tomllib
except ImportError:
    import tomli as tomllib
p = pathlib.Path('${LABS_DIR}/pyproject.toml')
print(tomllib.loads(p.read_text())['project']['version'])
")
LAB_HELPER_WHL="${REPO_ROOT}/wheels/mlsysbook_labs-${LAB_HELPER_VERSION}-py3-none-any.whl"
if [ ! -f "${LAB_HELPER_WHL}" ]; then
  echo "ERROR: Expected lab helper wheel not found after build: ${LAB_HELPER_WHL}" >&2
  exit 1
fi

BAD_HELPER_LABS=""
for lab in "${LABS_DIR}"/vol*/lab_*.py; do
  if grep -q "mlsysbook_labs" "${lab}" && ! grep -q "mlsysbook_labs-${LAB_HELPER_VERSION}-py3-none-any.whl" "${lab}"; then
    BAD_HELPER_LABS="${BAD_HELPER_LABS} ${lab}"
  fi
done
if [ -n "${BAD_HELPER_LABS}" ]; then
  echo "ERROR: Built lab helper wheel is ${LAB_HELPER_VERSION}, but these labs reference a different helper wheel:" >&2
  for lab in ${BAD_HELPER_LABS}; do
    echo "  - ${lab}" >&2
  done
  exit 1
fi
echo "Lab helper wheel version check passed: ${LAB_HELPER_VERSION}"

rm -rf "${LABS_DIR}/_wasm_build" "${LABS_DIR}/_build"
mkdir -p "${LABS_DIR}/_wasm_build/wheels"
cp "${MLSYSIM_DIR}"/dist/mlsysim-*.whl "${LABS_DIR}/_wasm_build/wheels/"
cp "${REPO_ROOT}"/wheels/mlsysbook_labs-*.whl "${LABS_DIR}/_wasm_build/wheels/"

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

rm -rf "${LABS_DIR}/_build/vol1" "${LABS_DIR}/_build/vol2"
mkdir -p "${LABS_DIR}/_build/vol1" "${LABS_DIR}/_build/vol2"
cp -R "${LABS_DIR}/_wasm_build/vol1/." "${LABS_DIR}/_build/vol1/"
cp -R "${LABS_DIR}/_wasm_build/vol2/." "${LABS_DIR}/_build/vol2/"
cp -r "${LABS_DIR}/_wasm_build/wheels" "${LABS_DIR}/_build/wheels"
# Duplicate wheels into vol directories to satisfy Pyodide worker relative paths
cp -r "${LABS_DIR}/_wasm_build/wheels" "${LABS_DIR}/_build/vol1/wheels"
cp -r "${LABS_DIR}/_wasm_build/wheels" "${LABS_DIR}/_build/vol2/wheels"
cp "${LABS_DIR}/site.webmanifest" "${LABS_DIR}/_build/site.webmanifest"

RELEASE_ID="${LABS_RELEASE_ID:-labs-v${LAB_HELPER_VERSION}-dev}"
RELEASE_HASH="$(
  cd "${REPO_ROOT}" && \
    python3 scripts/version/release.py compute-hash \
      --paths labs mlsysim/mlsysim \
      --exclude _build _wasm_build .quarto __pycache__ .pytest_cache node_modules "*.pyc" release-manifest.json
)"
METADATA_JSON="$(
  LAB_COUNT="${expected}" \
  MLSYSIM_VERSION="${BUILT_VERSION}" \
  LAB_HELPER_VERSION="${LAB_HELPER_VERSION}" \
  python3 - <<'PY'
import json
import os

print(json.dumps({
    "labCount": int(os.environ["LAB_COUNT"]),
    "volumes": 2,
    "mlsysimVersion": os.environ["MLSYSIM_VERSION"],
    "labHelperVersion": os.environ["LAB_HELPER_VERSION"],
    "releaseChannel": "dev",
    "runtime": "marimo-wasm",
}))
PY
)"
python3 "${REPO_ROOT}/scripts/version/release.py" emit-manifest \
  --output "${LABS_DIR}/_build/release-manifest.json" \
  --project labs \
  --tier B \
  --release-id "${RELEASE_ID}" \
  --release-hash "${RELEASE_HASH}" \
  --schema-version "1" \
  --metadata "${METADATA_JSON}" >/dev/null

# The production path is /labs/release-manifest.json. Keeping this mirror lets
# local previews served from _build/ exercise the same absolute manifest URL.
mkdir -p "${LABS_DIR}/_build/labs"
cp "${LABS_DIR}/_build/release-manifest.json" "${LABS_DIR}/_build/labs/release-manifest.json"

wasm_count="$(find "${LABS_DIR}/_build/vol1" "${LABS_DIR}/_build/vol2" -name "index.html" | wc -l | tr -d ' ')"
echo "WASM notebooks in final build: ${wasm_count}"
if [ "${wasm_count}" -ne "${expected}" ]; then
  echo "ERROR: Expected ${expected} WASM notebooks in build output, got ${wasm_count}" >&2
  exit 1
fi

missing_routes=""
for vol in vol1 vol2; do
  for lab in "${LABS_DIR}/${vol}"/lab_*.py; do
    name="$(basename "${lab}" .py)"
    route="${LABS_DIR}/_build/${vol}/${name}/index.html"
    if [ ! -f "${route}" ]; then
      missing_routes="${missing_routes} ${vol}/${name}"
    fi
  done
done
if [ -n "${missing_routes}" ]; then
  echo "ERROR: Missing direct WASM routes in build output:${missing_routes}" >&2
  exit 1
fi

test -f "${LABS_DIR}/_build/index.html"
test -f "${LABS_DIR}/_build/release-manifest.json"
test -f "${LABS_DIR}/_build/site.webmanifest"
test -f "${LABS_DIR}/_build/assets/images/favicon.svg"
test -f "${LABS_DIR}/_build/assets/images/social-card.svg"
echo "Labs site build complete: ${LABS_DIR}/_build"
