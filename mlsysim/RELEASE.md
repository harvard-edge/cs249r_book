# MLSys·im Release Runbook

Releases are **automated** via the `mlsysim-pypi-publish.yml` GitHub Actions
workflow. Publishing happens when a tag matching `mlsysim-v*` is pushed to
origin. The workflow authenticates to PyPI via **Trusted Publishing (OIDC)** —
no PyPI API token is stored in the repo or in GitHub Secrets.

## Prepare and validate the candidate

Prepare release changes in an isolated worktree. Align `pyproject.toml`,
`mlsysim/__init__.py`, `CITATION.cff`, and the top `CHANGELOG.md` entry. Install
the development environment described in [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
make check
make publication-check
make build
python -m twine check dist/*
```

Install the built wheel in a fresh environment outside the repository and run
an import, registry lookup, solver call, and `mlsysim --help`. This catches
missing packaged data and imports hidden by editable installation. Test the
source distribution too when changing packaging rules.

Before tagging, run the downstream checks in CONTRIBUTING.md. Examine changed
scenario outputs and identify affected textbook or paper claims. Prose and
figure corrections require their own review; a passing package suite does not
clear that gate. The paper's generated-value freshness check must be resolved
before a coordinated package/documentation release.

## Compare downstream numerical outputs

Use the same trusted consumer checkout and interpreter for both snapshots. Only
change the selected MLSysIM package directory. This isolates implementation
changes from edited chapter examples. The package directory must contain the
`mlsysim/` import package; it may come from a baseline checkout or a Git archive.
Never use a snapshot captured against different chapter sources as a baseline.

From the repository root, with the baseline package at `/path/to/baseline/mlsysim`:

```bash
mlsysim/.venv/bin/python mlsysim/scripts/downstream_impact.py snapshot \
  --package /path/to/baseline/mlsysim --output /tmp/impact-before.json
mlsysim/.venv/bin/python mlsysim/scripts/downstream_impact.py snapshot \
  --package mlsysim --output /tmp/impact-after.json
mlsysim/.venv/bin/python mlsysim/scripts/downstream_impact.py compare \
  /tmp/impact-before.json /tmp/impact-after.json --output /tmp/impact-review.json
```

The default scope discovers Python consumers in all textbook volumes and runs
the paper's deterministic scenario sections. Each consumer runs in a separate
process with a fresh snapshot bytecode cache; the tool checks the imported package path and captures source hashes,
raw accessible numerical exports, formatted inline values, and nearby prose.
A raw change remains visible even when rounding leaves the printed text unchanged.
No prose is rewritten. The JSON report names outputs and explanatory passages
that need a separate editorial decision.

Exit status `0` means the declared numerical comparison is unchanged, `1` means
changed outputs or skipped cells require review, and `2` means the comparison is
incomplete (for example, source drift, missing coverage, or execution errors).
Snapshot commands also fail on execution errors. Do not treat an incomplete run
as an unchanged result, and do not suppress a changed number by repinning a test.
Freeze package and consumer edits while collecting snapshots; concurrent source
changes invalidate the run. Retain both snapshots with the report.

For a narrow iteration, repeat `--glob 'books/vol1/13_model_serving/*.qmd'` and use
`--no-paper` when appropriate; record that limited scope explicitly. The final
release review must include every affected consumer. This tool does not execute
browser labs or slides, does not render publications, suppresses Matplotlib save/show calls,
and records `eval:false` cells as unexecuted. Its output cannot approve plots,
layout, changed bottleneck explanations, or material outside the declared scope.
Run the existing lab/Binder checks and inspect affected rendered publications
separately before publication. Local timings have their own publication receipt
gate and are excluded from the numerical scenario comparison.

## Publish the reviewed release

After the reviewed changes have been merged to `dev` and release approval has
been given, tag the reviewed commit and push only that tag. For example, for
the candidate prepared here:

```bash
git tag -a mlsysim-v0.3.1 <reviewed-dev-commit> -m "MLSys·im 0.3.1"
git push origin mlsysim-v0.3.1
```

Pushing the tag publishes externally. Do not use it as a packaging test.

From there, the workflow does everything:

1. **Verify** — tag format, version coherence across `pyproject.toml` /
   `__init__.py` / `CITATION.cff`, CHANGELOG entry present, tag reachable
   from dev.
2. **Test** — full pytest suite in parallel across every supported Python
   version (3.10, 3.11, 3.12, 3.13). Catches cross-version issues
   *before* the wheel ships.
3. **Build** — `python -m build` → wheel + sdist; `twine check` on both.
4. **Publish to PyPI** — via `pypa/gh-action-pypi-publish` + OIDC, with
   PEP 740 attestations (cryptographic provenance).
5. **Verify from PyPI** — post-publish smoke test: `pip install` from the
   public PyPI (with CDN propagation retry), import check, CLI smoke.
   Closes the gap between "upload accepted" and "user install works."
6. **GitHub Release** — creates the release, attaches wheel + sdist, uses
   `RELEASE_NOTES_<version>.md` as the body if present.
7. **Docs redeploy** — dispatches `mlsysim-publish-live.yml` on `dev` so the
   docs site reflects the new version.

Monitor the run: <https://github.com/harvard-edge/cs249r_book/actions/workflows/mlsysim-pypi-publish.yml>

## Pre-release checklist (before tagging)

Run this locally to catch the easy failures before the workflow does:

- [ ] `cd mlsysim && make check` → all package checks pass
- [ ] `make publication-check` → displayed paper values and timing receipts pass
- [ ] Baseline/candidate downstream report reviewed; every changed claim or incomplete check has an explicit disposition
- [ ] Versions aligned: all four places read the same `X.Y.Z`
  ```bash
  rg '^version = ' mlsysim/pyproject.toml
  rg '^__version__ = ' mlsysim/mlsysim/__init__.py
  rg '^version:' mlsysim/CITATION.cff
  head -3 mlsysim/CHANGELOG.md
  ```
- [ ] CHANGELOG top entry is the version about to ship
- [ ] Optional: `mlsysim/RELEASE_NOTES_<version>.md` written for the GH Release body
- [ ] CLI smoke: `mlsysim eval Llama3_8B H100 --batch-size 32` returns a scorecard
- [ ] Docs render cleanly: `cd mlsysim/docs && quarto render` — no
  `Unable to resolve link target` warnings

If anything fails, fix on a PR to dev and merge before tagging. The workflow
will re-run these checks, but catching them locally saves 5 minutes per
attempt.

## Trusted Publishing — one-time setup

If the workflow fails with an OIDC error, Trusted Publishing has not been
configured on pypi.org. Do this once:

1. Sign in to <https://pypi.org/> as a maintainer of the `mlsysim` project.
2. Go to <https://pypi.org/manage/account/binder/>.
3. Click **Add a new pending publisher** (or **Manage** for existing ones).
4. Fill in:

   | Field       | Value                      |
   |-------------|----------------------------|
   | PyPI Project name | `mlsysim`            |
   | Owner       | `harvard-edge`             |
   | Repository name  | the name in `harvard-edge/cs249r_book` |
   | Workflow name    | `mlsysim-pypi-publish.yml` |
   | Environment name | `pypi-mlsysim`        |

5. Save. No token is ever generated; GitHub's OIDC provider attests the
   workflow identity at publish time and PyPI trusts that attestation.

This setup is per-project and per-workflow. It stays in place across workflow
runs indefinitely; only re-do it if the workflow filename or environment name
changes.

## Post-release verification

From a clean venv (CI already ran this, but a human spot-check catches UX bugs):

```bash
python -m venv /tmp/release-verify && source /tmp/release-verify/bin/activate
pip install mlsysim==<just-released-version>
python -c "import mlsysim; print('OK', mlsysim.__version__)"
mlsysim eval Llama3_8B H100 --batch-size 32
deactivate
```

Open <https://mlsysbook.ai/mlsysim/> in an incognito window; confirm:

- Version number on the site matches
- Navbar shows `MLSys·im` (mixed case, not `MLSys·IM`)
- Footer shows `Code: Apache-2.0 · Docs: CC-BY-NC-SA 4.0`
- Getting Started and Tutorials still load

## Announce (optional)

- Bump `mlsysim/docs/config/announcement.yml` banner if the release is
  user-visible (major features, breaking changes).
- Cross-post to the textbook newsletter / course channels.

## Rollback

You cannot re-upload a PyPI version, even after deleting it. If a release
has a critical bug:

1. **Yank** the bad version on pypi.org (Manage → Versions → Yank).
   This hides it from `pip install mlsysim` but keeps existing pins working.
2. Fix the bug on a PR to `dev`, bump to the next patch version
   (`X.Y.Z+1`), merge.
3. Tag `mlsysim-vX.Y.Z+1` and push — the workflow ships the fix.

Never force-push or amend a tag that's been pushed. Tags on origin are
immutable release markers.

---

## Manual fallback — only if the workflow is broken

If the workflow itself has a bug that prevents automated release and you
*must* ship, the legacy manual steps still work (your `~/.pypirc` with a
PyPI token is the required credential path here). Fix the workflow in a
follow-up PR; don't normalize the manual path.

```bash
# From a clean checkout of the tagged commit:
cd mlsysim
make clean && make build
twine check dist/*
twine upload dist/*                  # requires ~/.pypirc with scoped token
gh release create mlsysim-vX.Y.Z \
  --repo harvard-edge/cs249r_book \
  --title "MLSys·im X.Y.Z" \
  --notes-file RELEASE_NOTES_X.Y.Z.md \
  dist/mlsysim-X.Y.Z-py3-none-any.whl \
  dist/mlsysim-X.Y.Z.tar.gz
gh workflow run mlsysim-publish-live.yml -R harvard-edge/cs249r_book --ref dev
```
