# Versioning

How releases work across the MLSysBook monorepo. One pattern for all
publishable artifacts; one source of truth per release; one URL the
operator visits to ship one.

## Why this exists

Before this convention landed, every project had its own version story:

- StaffML built every publish as `--release-id publish-live` (a constant
  string), and the manifest drifted to `0.1.2-dev` while
  `releases/0.1.0/release.json` said `0.1.0`. Citations couldn't be
  trusted.
- TinyTorch hand-edited the version in 6 files per release; one missed
  edit could silently downgrade the source.
- Book had per-volume tags but no cross-coordination — Vol I and Vol II
  could disagree about which release shipped together.
- MLSYSIM had a PyPI version, a paper, and a docs site that all claimed
  different things at any given time.
- Kits/Labs/Instructors had no versions at all — just commit SHAs.

The unified convention fixes the operational problem ("did I ship the
right thing? what's deployed right now?") and the citation problem
("v0.1.0 means *exactly these bytes*, forever").

## What every release produces

Each `<project>-publish-live` workflow run produces:

1. **A git tag** — `<project>-v<release_id>`, e.g. `staffml-v0.1.1`.
2. **A `release-manifest.json`** at the deploy URL — e.g.
   `https://mlsysbook.ai/staffml/release-manifest.json`. Cacheable,
   readable by anyone, parseable by tools.
3. **A draft GitHub Release** — for the human-facing changelog with
   auto-generated commit list (Tier A also runs AI-enhanced notes).
4. **A footer pill on the live site** — small inline "v0.1.1 · Apr 28,
   2026" element. Click to copy hash. Best-effort chrome.

Tier A projects additionally produce:

5. **A `release_hash`** — Merkle-style SHA-256 over input bytes,
   recorded in the manifest. This is the citation anchor: a paper
   referencing "MLSysBook v0.1.1 (hash 6883e85)" is now reproducible.

## How to publish (operator)

1. Go to the project's "Publish (Live)" workflow in GitHub Actions.
2. Click "Run workflow" and fill in:
   - **release_type**: `patch` (small fixes), `minor` (new content),
     `major` (breaking changes). Default `patch`.
   - **description**: One-line summary. Becomes the release-notes title.
   - **site_only**: Check this for CSS/copy-only redeploys that
     should NOT bump the release_id (citation integrity demands a
     given version maps to fixed bytes — re-tagging existing releases
     is forbidden).
   - **explicit_version**: For non-incremental jumps (e.g. 0.1.x →
     0.10.0 alongside a coordinated launch). Leave blank to auto-bump.
   - **confirm**: Type `PUBLISH`. The workflow refuses to proceed if
     this isn't exact — stops accidental clicks.
3. Wait for the workflow to complete. The draft GitHub Release will
   appear at `https://github.com/<repo>/releases`. Review the auto-
   generated notes, then publish.

## How to verify a deployed version

```bash
# What release is live right now?
curl -s https://mlsysbook.ai/staffml/release-manifest.json | jq .

# What's in release 0.1.1?
gh release view staffml-v0.1.1
cat releases/staffml-0.1.1/release.json | jq .

# Does the deployed manifest match what's tagged?
curl -s https://mlsysbook.ai/staffml/release-manifest.json | jq -r .releaseHash
git show staffml-v0.1.1:releases/staffml-0.1.1/release.json | jq -r .release_hash
```

If those two hashes differ, something between tag and deploy went
wrong — file an issue.

## Who lives in which tier

| Project | Tier | Why |
|---|---|---|
| StaffML | A | Citable question bank; authors will reference v X.Y.Z in papers |
| TinyTorch | A | Educational framework; cited in syllabi and papers |
| Book Vol I | A | Textbook, multiple per-volume tag tracks (vol1-v*) |
| Book Vol II | A | Textbook, separate tag track (vol2-v*) |
| MLSYSIM | A | Site identity binds to PyPI + paper, all citable |
| Kits | B | Hardware deployment labs, iterate fast, not formally cited |
| Labs | B | Marimo notebooks, evolve constantly |
| Instructors | B | Instructor guide, lower citation stakes |

Tier A and Tier B share the workflow UX. They differ in:

- **Hash detail**: Tier A includes a per-file `files: [{path, hash}]`
  index in `release.json` (Merkle-ish: lets a consumer verify a single
  question without downloading the whole corpus). Tier B uses a flat
  SHA-256.
- **Release notes**: Tier A runs AI-enhanced summarization. Tier B
  uses plain auto-generated commit lists.

## Architecture in one paragraph

`scripts/version/release.py` is the canonical implementation of every
versioning operation: hash a directory, compute next release_id, emit
a release.json, emit a build-time manifest. The reusable workflow
`.github/workflows/_release-prepare.yml` validates `confirm`, computes
the new release_id from prior tag + bump, and outputs values the
caller workflow uses to drive its own build. Each project's
`<project>-publish-live.yml` calls `_release-prepare.yml` first, runs
its existing build with the computed release_id, emits a manifest
into the build output (so it deploys at the canonical URL), and then
tags + creates a release. The `shared/release/release-pill.html`
fragment fetches the manifest at runtime and renders the footer pill;
each project's Quarto config sets a `<meta name="release-manifest">`
tag so the snippet finds the right URL.

## Contract reference

| Field | Source | Notes |
|---|---|---|
| `releaseId` | `_release-prepare.yml` output | Bare semver, no prefix |
| `releaseHash` | `release.py compute-hash` | 64 hex chars; Merkle-ish (Tier A) or flat SHA-256 (Tier B) |
| `schemaVersion` | Manifest emitter caller | Project-internal; rarely changes |
| `tier` | Manifest emitter caller | `A` or `B` |
| `project` | Manifest emitter caller | Short identifier |
| `buildDate` | Manifest emitter | UTC ISO 8601 — set at emit time |

The full schema is in `scripts/version/schema.json`.

## Backfill

Pre-cutover releases (everything tagged before this convention landed)
do NOT have a `releases/<id>/release.json` artifact. They keep their
existing tags and behave normally; only releases ≥ this PR get the
full treatment. Backfilling historical releases is a separate, optional
follow-up task.

## Out of scope (today)

- **Periodic-Table** — no publish workflow exists; we'd be adding
  versioning to a project that doesn't deploy. Establish publishing
  first.
- **Reusable build orchestration** — the reusable workflow only does
  prepare. Each project keeps its own build/test/deploy because those
  steps are project-specific (Quarto vs Next.js vs Marimo). Trying to
  generalize the build itself produced too-thin or too-rigid abstractions
  in earlier drafts.
- **PyPI version unification** — MLSYSIM's PyPI version comes from
  `pyproject.toml` and remains the canonical source for the package.
  The site's release identity is a separate (compatible) track. The
  paper's identity rides on the site release.
- **Cross-project release coordination** — there's no "stamp all of
  MLSysBook with the same release_id" path. Each project bumps
  independently. The book's per-volume coordination is the only
  intentional exception.

## Files at a glance

```
docs/VERSIONING.md                              ← this file
scripts/version/release.py                      ← Python helpers + CLI
scripts/version/schema.json                     ← JSON Schema for release.json
shared/release/README.md                        ← contract documentation
shared/release/release-pill.html                ← footer snippet
shared/release/release-card.html                ← about-page snippet
.github/workflows/_release-prepare.yml          ← reusable workflow

# Per-project: each <project>-publish-live.yml calls _release-prepare,
# emits a manifest before deploy, tags + drafts a release after.
.github/workflows/staffml-publish-live.yml
.github/workflows/tinytorch-publish-live.yml
.github/workflows/book-publish-live.yml
.github/workflows/mlsysim-publish-live.yml
.github/workflows/kits-publish-live.yml
.github/workflows/labs-publish-live.yml
.github/workflows/instructors-publish-live.yml

# Per-project Quarto/Next config: meta tag + pill include
interviews/staffml/src/components/Footer.tsx    ← StaffML uses build-time bake
tinytorch/quarto/_quarto.yml
book/quarto/config/_quarto-html-vol1.yml
book/quarto/config/_quarto-html-vol2.yml
mlsysim/docs/config/_quarto-html.yml
kits/config/_quarto-html.yml
labs/config/_quarto-html.yml
instructors/_quarto.yml
```
