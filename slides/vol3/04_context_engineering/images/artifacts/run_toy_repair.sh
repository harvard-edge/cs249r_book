#!/usr/bin/env bash
# Real runs behind three slides of Lecture 4 (Context Engineering).
# A toy connection pool, written for this lecture, with the chapter's
# missed-notification defect at r17 and the agent's patch at r18.
#   usage: bash run_toy_repair.sh <python> <empty workdir>
# Writes pytest_r17.txt, pytest_r18.txt, git_apply_r18.txt beside this script.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$1"
W="$2"
mkdir -p "$W/src" "$W/tests"
cd "$W" || exit 1
git init -q
git config user.email lecture@example.org
git config user.name lecture
git config core.hooksPath /dev/null
touch src/__init__.py tests/__init__.py
cp "$HERE/test_pool.py" tests/test_pool.py
cp "$HERE/pool_r17.py" src/pool.py
git add -A && git commit -qm r17 && git tag r17
"$PY" -m pytest -q -p no:cacheprovider tests/test_pool.py > "$HERE/pytest_r17.txt" 2>&1
cp "$HERE/pool_r18.py" src/pool.py
git commit -qam r18 && git tag r18
"$PY" -m pytest -q -p no:cacheprovider tests/test_pool.py > "$HERE/pytest_r18.txt" 2>&1
git apply --check -v "$HERE/stale_from_r17.patch" > "$HERE/git_apply_r18.txt" 2>&1
echo "workdir: $W"
