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

# Check chapter ownership before compiling goals, inventory, or browser wheels.
python3 "${LABS_DIR}/tools/check_learning_contracts.py"
python3 "${LABS_DIR}/tools/build_learning_paths.py"
python3 "${LABS_DIR}/tools/build_learning_paths.py" --check
python3 "${LABS_DIR}/tools/build_lab_inventory.py"
python3 "${LABS_DIR}/tools/build_lab_inventory.py" --check

python3 -m build --wheel "${MLSYSIM_DIR}"
python3 -m build --wheel "${LABS_DIR}" --outdir "${LABS_DIR}/wheels"

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
mkdir -p "${LABS_DIR}/wheels"
cp "${BUILT_WHL}" "${LABS_DIR}/wheels/"

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
LAB_HELPER_WHL="${LABS_DIR}/wheels/mlsysbook_labs-${LAB_HELPER_VERSION}-py3-none-any.whl"
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

# Only clean disposable build children; never recursively remove a checkout path.
python3 - "${LABS_DIR}" <<'PY_CLEAN'
from pathlib import Path
import shutil
import sys
labs = Path(sys.argv[1]).resolve()
for name in ("_wasm_build", "_build"):
    output = labs / name
    if output.is_symlink():
        raise SystemExit(f"Refusing symlinked build output: {output}")
    if output.exists():
        shutil.rmtree(output)
PY_CLEAN
mkdir -p "${LABS_DIR}/_wasm_build/wheels"
cp "${MLSYSIM_DIR}"/dist/mlsysim-*.whl "${LABS_DIR}/_wasm_build/wheels/"
cp "${LABS_DIR}"/wheels/mlsysbook_labs-*.whl "${LABS_DIR}/_wasm_build/wheels/"

expected=0
exported=0
failed=""

mapfile_compat_routes="$(python3 "${LABS_DIR}/tools/build_lab_shells.py" list)"
while IFS= read -r route; do
    vol="${route%%/*}"
    lab="${LABS_DIR}/${route}.py"
    name="$(basename "${lab}" .py)"
    expected=$((expected + 1))
    echo "Exporting ${vol}/${name}..."
    if marimo export html-wasm "${lab}" \
      -o "${LABS_DIR}/_wasm_build/${vol}/${name}/app/index.html" \
      --mode run \
      --no-show-code; then
      # Marimo bundles an assistant instruction file with its static assets.
      # Keep AI configuration out of the public site artifact.
      python3 - "${LABS_DIR}/_wasm_build/${vol}/${name}/app/CLAUDE.md" <<'PY_ASSET'
from pathlib import Path
import sys
Path(sys.argv[1]).unlink(missing_ok=True)
PY_ASSET
      exported=$((exported + 1))
    else
      failed="${failed} ${vol}/${name}"
    fi
done <<< "${mapfile_compat_routes}"

echo "WASM export summary: ${exported}/${expected} notebooks exported"

# marimo writes its stock AI-assistant notes (CLAUDE.md) into every export.
# They are not part of a lab and must not be published.
find "${LABS_DIR}/_wasm_build" -name CLAUDE.md -delete

if [ -n "${failed}" ]; then
  echo "ERROR: Failed exports:${failed}" >&2
  exit 1
fi

# Transient QMD routes inherit the actual shared Quarto site metadata.
# Cleanup runs after failures as well; only marker-bearing generated files unlink.
python3 "${LABS_DIR}/tools/build_lab_shells.py" prepare
trap 'python3 "${LABS_DIR}/tools/build_lab_shells.py" cleanup' EXIT
(cd "${LABS_DIR}" && quarto render && touch _build/.nojekyll)
python3 "${LABS_DIR}/tools/build_lab_shells.py" cleanup
trap - EXIT

# Install full-page apps and launch routes; preserve Volume IV hardware pages.
cp -R "${LABS_DIR}/_wasm_build/." "${LABS_DIR}/_build/"
python3 "${LABS_DIR}/tools/build_lab_shells.py" install
python3 - "${LABS_DIR}/_build" "${mapfile_compat_routes}" <<'PY_WHEELS'
from pathlib import Path
import shutil
import sys
output = Path(sys.argv[1])
for route in sys.argv[2].splitlines():
    # Relative micropip URLs are resolved from worker/app locations differently
    # across Marimo releases. Supply each supported ancestor, never public CDNs.
    for destination in (output / route.split("/")[0] / "wheels", output / route / "wheels", output / route / "app/wheels"):
        shutil.copytree(output / "wheels", destination, dirs_exist_ok=True)
PY_WHEELS
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
  LAB_VOLUMES="$(printf '%s\n' "${mapfile_compat_routes}" | cut -d/ -f1 | sort -u | wc -l | tr -d ' ')" \
  MLSYSIM_VERSION="${BUILT_VERSION}" \
  LAB_HELPER_VERSION="${LAB_HELPER_VERSION}" \
  python3 - <<'PY'
import json
import os

print(json.dumps({
    "labCount": int(os.environ["LAB_COUNT"]),
    "volumes": int(os.environ["LAB_VOLUMES"]),
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

while IFS= read -r route; do
  test -f "${LABS_DIR}/_build/${route}/index.html" || { echo "Missing Quarto shell: ${route}" >&2; exit 1; }
  test -f "${LABS_DIR}/_build/${route}/app/index.html" || { echo "Missing Marimo app: ${route}" >&2; exit 1; }
done <<< "${mapfile_compat_routes}"
echo "Verified ${expected} full-page interactive lab routes"

test -f "${LABS_DIR}/_build/index.html"
test -f "${LABS_DIR}/_build/release-manifest.json"
test -f "${LABS_DIR}/_build/site.webmanifest"
test -f "${LABS_DIR}/_build/assets/images/favicon.svg"
test -f "${LABS_DIR}/_build/assets/images/social-card.svg"
echo "Labs site build complete: ${LABS_DIR}/_build"
