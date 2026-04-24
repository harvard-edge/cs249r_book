# MLSys·im Release Runbook

Releases are **automated** via the `mlsysim-pypi-publish.yml` GitHub Actions
workflow. Publishing happens when a tag matching `mlsysim-v*` is pushed to
origin. The workflow authenticates to PyPI via **Trusted Publishing (OIDC)** —
no PyPI API token is stored in the repo or in GitHub Secrets.

## Happy path — the 3-step release

Prerequisite: your changes must already be merged to `dev`, with the version
bumped in `pyproject.toml`, `mlsysim/__init__.py`, `CITATION.cff`, and an entry
added to `CHANGELOG.md`.

```bash
# 1. Move to merged dev
git checkout dev
git pull --ff-only origin dev

# 2. Tag the release (annotated, prefixed)
git tag -a mlsysim-v0.1.2 -m "MLSys·im 0.1.2"

# 3. Push the tag → the workflow fires automatically
git push origin mlsysim-v0.1.2
```

From there, the workflow does everything:

1. **Verify** — tag format, version coherence across `pyproject.toml` /
   `__init__.py` / `CITATION.cff`, CHANGELOG entry present, tag reachable
   from dev.
2. **Test** — full pytest suite in a clean Python 3.11 container.
3. **Build** — `python -m build` → wheel + sdist; `twine check` on both.
4. **Publish to PyPI** — via `pypa/gh-action-pypi-publish` + OIDC.
5. **GitHub Release** — creates the release, attaches wheel + sdist, uses
   `RELEASE_NOTES_<version>.md` as the body if present.
6. **Docs redeploy** — dispatches `mlsysim-publish-live.yml` on `dev` so the
   docs site reflects the new version.

Monitor the run: <https://github.com/harvard-edge/cs249r_book/actions/workflows/mlsysim-pypi-publish.yml>

## Pre-release checklist (before tagging)

Run this locally to catch the easy failures before the workflow does:

- [ ] `cd mlsysim && pytest tests/ -q` → 0 failures
- [ ] Versions aligned: all four places read the same `X.Y.Z`
  ```bash
  grep -E '^version = ' mlsysim/pyproject.toml
  grep -E '^__version__ = ' mlsysim/mlsysim/__init__.py
  grep -E '^version:' mlsysim/CITATION.cff
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
2. Go to <https://pypi.org/manage/account/publishing/>.
3. Click **Add a new pending publisher** (or **Manage** for existing ones).
4. Fill in:

   | Field       | Value                      |
   |-------------|----------------------------|
   | PyPI Project name | `mlsysim`            |
   | Owner       | `harvard-edge`             |
   | Repository name  | `cs249r_book`         |
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
