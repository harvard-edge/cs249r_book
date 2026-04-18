# MLSys·im Release Runbook

This is the operational runbook for cutting an MLSys·im release. Use it verbatim — every step is intended to be copy-paste-able.

The current version under `mlsysim/pyproject.toml` is the source of truth; bump it before tagging.

## 0. Pre-flight checklist

Run from the **`dev`** branch after the release-prep PR has merged.

- [ ] Tests green: `cd mlsysim && pytest tests/ -q` reports the expected pass count (367 as of 0.1.0) and `0 failed`.
- [ ] Versions aligned: `grep -E '^(version|__version__|date-released)' mlsysim/pyproject.toml mlsysim/CITATION.cff mlsysim/__init__.py` and `head -3 mlsysim/CHANGELOG.md` all read the same `X.Y.Z` and date.
- [ ] Changelog updated: top entry of `mlsysim/CHANGELOG.md` is the version about to ship.
- [ ] Release notes written: `mlsysim/RELEASE_NOTES_<version>.md` exists.
- [ ] Docs build cleanly: `cd mlsysim/docs && quarto render` produces no `Unable to resolve link target` warnings.
- [ ] CLI smoke: `mlsysim eval Llama3_8B H100 --batch-size 32` returns a scorecard.

If any of these fail, **stop and fix on a PR** before continuing.

## 1. Tag the release

Tags follow the pattern `mlsysim-vX.Y.Z` (the `mlsysim-` prefix avoids collision with TinyTorch and slide tags in the same monorepo).

```bash
git checkout dev
git pull --ff-only origin dev
git tag -a mlsysim-v0.1.0 -m "MLSys·im 0.1.0 — initial release"
git push origin mlsysim-v0.1.0
```

## 2. Build the distribution

```bash
cd mlsysim
make clean
make build           # → dist/mlsysim-X.Y.Z-py3-none-any.whl + dist/mlsysim-X.Y.Z.tar.gz
```

Sanity-check the wheel before publishing:

```bash
python -m venv /tmp/wheel-test && source /tmp/wheel-test/bin/activate
pip install dist/mlsysim-*.whl
python -c "import mlsysim; print(mlsysim.__version__)"
mlsysim eval Llama3_8B H100 --batch-size 32
deactivate && rm -rf /tmp/wheel-test
```

## 3. Publish to PyPI

```bash
cd mlsysim
pip install --upgrade twine
twine check dist/*
twine upload dist/*           # uses ~/.pypirc credentials
```

For a dry-run upload to TestPyPI:

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ mlsysim==X.Y.Z
```

## 4. Deploy the docs site

The docs workflow is `workflow_dispatch` only — trigger it manually after the tag is in place.

```bash
gh workflow run mlsysim-publish-live.yml -R harvard-edge/cs249r_book --ref dev
gh run watch -R harvard-edge/cs249r_book
```

When the run completes, visit <https://mlsysbook.ai/mlsysim/> and confirm:

- Hero example output matches the engine (no stale numbers).
- Navbar shows `MLSys·im` (mixed case, not `MLSys·IM`).
- API Reference loads.

## 5. Cut the GitHub Release

```bash
gh release create mlsysim-v0.1.0 \
  --repo harvard-edge/cs249r_book \
  --title "MLSys·im 0.1.0" \
  --notes-file mlsysim/RELEASE_NOTES_0.1.0.md \
  mlsysim/dist/mlsysim-0.1.0-py3-none-any.whl \
  mlsysim/dist/mlsysim-0.1.0.tar.gz
```

## 6. Final verification

From a clean machine (or a fresh container):

```bash
python -m venv /tmp/release-verify && source /tmp/release-verify/bin/activate
pip install mlsysim==0.1.0
python -c "import mlsysim; print('OK', mlsysim.__version__)"
mlsysim eval Llama3_8B H100 --batch-size 32
```

Open <https://mlsysbook.ai/mlsysim/> in an incognito window and click through:

- Getting Started
- Tutorials → Hello, Roofline
- API Reference → Solvers
- Footer links resolve

## 7. Announce (optional)

- Bump the announcement banner in `mlsysim/docs/config/announcement.yml` if you want to flag the new release on every page for a few weeks.
- Cross-post a short note to the textbook newsletter / course channels.

---

## Rollback

If a critical bug ships:

```bash
# Yank the bad version from PyPI (cannot be re-uploaded under the same number)
twine upload --skip-existing dist/*    # never overwrite — bump and re-release
pip install mlsysim==<previous>        # confirm the previous version still installs
```

Then publish `X.Y.(Z+1)` with the fix following the same runbook. Never amend a tag that has been pushed.
