# Release-Prep Handoff

This worktree (`MLSysBook-release-prep` / branch
`release-prep/staged-rollout-foundation`) contains the foundation for the
staged rollout of the new mlsysbook.ai ecosystem properties (Volumes I & II,
TinyTorch, Labs, Hardware Kits, MLSys·im, Lecture Slides, Instructor Hub,
StaffML, and the unified landing page).

Nothing in this branch ships anything live by itself. Everything is opt-in
plumbing that the per-property launch PRs will turn on as each subsite
graduates from the dev mirror to mlsysbook.ai.

---

## How this branch is meant to be merged

The 18 commits on this branch are organized into **five logical groupings**
that are intentionally independently reviewable, so they can be split into
five PRs against `dev`. Each grouping is self-contained: any one of them
can be merged without the others.

The recommended PR order is the order shown below, but only PR-1
(safety-net) is a hard prerequisite for the others. PR-3 / PR-4 / PR-5 can
ship in any sequence.

---

## PR-1 — Safety net (link checking + publish guards + nightly link-rot)

**Commits:** `fb16b83c1`, `0371cc6dd`, `550bbed0b`

The "no broken link can survive long enough to embarrass us in front of
readers" tier. Three layers, each cheap on its own:

- **Tier 1 — pre-commit (instant, local).** New
  `shared/scripts/check-internal-links.py` runs against staged `.md`/`.qmd`
  files only, validates internal links + anchors, skips fenced code blocks
  and inline code (so TikZ snippets etc. don't false-positive), and is
  wired into `.pre-commit-config.yaml` with an `exclude` list for known
  legacy / auto-generated paths.

- **Tier 2 — per-subsite CI (every PR + every dev push).** Generalized
  `.github/workflows/infra-link-check.yml` to take site-specific args
  (`include_pattern`, `lycheeignore_path`, `accept_status`,
  `fail_on_broken`). Every `*-validate-dev.yml` now invokes it as a
  `check-links` job. Created `site-validate-dev.yml` and
  `staffml-validate-dev.yml` for the two subsites that lacked validate
  workflows. Universal exclusions live in `shared/config/.lycheeignore`.

- **Publish guard.** New `.github/workflows/infra-publish-guard.yml`
  blocks every `*-publish-live.yml` from running unless the latest
  `*-validate-dev.yml` run on `dev` is green and recent (configurable
  `max_age_minutes`, default 60). Wired as the first job of each publish
  workflow.

- **Tier 3 — nightly link-rot tracker.** New
  `.github/workflows/infra-link-rot-nightly.yml` sweeps every subsite at
  04:30 UTC and aggregates findings into a single sticky GitHub issue
  (label `link-rot`, title pinned). Each run replaces the issue body and
  appends a comment, so the issue is the always-current dashboard.

**Risk surface:** Tier 1 + Tier 2 are non-blocking by default
(`fail_on_broken: false`) so existing baseline failures don't immediately
gate merges; we can flip the flag per-subsite as backlogs are cleared.
Publish guards CAN block on green dev — that's the whole point.

---

## PR-2 — Theme / sidebar / dev-mirror visual polish

**Commits:** `2d2fa2edd`, `bafc86896`, `ce180eb91`, `d119b2e2b`

Visual / cross-site UX work prerequisite to launch:

- **`fix(dev-mirror)`** — `rewrite-dev-urls.sh` was hard-coding the dev-
  side prefix to `../`, which broke nested subsites like `book/vol1`
  and `book/vol2` (their cross-site nav links pointed one level too
  deep). Fixed to compute the prefix from the actual dev-side path
  depth. Also fixed an associative-array key check that wasn't portable
  to macOS bash 3.2.

- **Per-volume announcement bars.** Split the shared `announcement.yml`
  into `announcement-vol1.yml` (Harvard Crimson tint, Vol I copy) and
  `announcement-vol2.yml` (ETH Blue tint, Vol II copy). The old shared
  file is left as a deprecated no-op with a pointer comment so any
  external reference still resolves. Wired into the volume-specific
  Quarto configs.

- **Cross-site dark-mode persistence.** New `shared/scripts/theme-persist.js`
  inlined into `shared/config/site-head.html` so it executes
  synchronously in `<head>` (no FOUC). Reads `quarto-color-scheme` from
  `localStorage`, falls back to OS preference, applies `data-bs-theme` /
  `data-quarto-color-scheme` / `html.style.colorScheme`. Listens for
  cross-tab `storage` events. Bridged into StaffML
  (`interviews/staffml/public/theme-bootstrap.js` and
  `src/components/ThemeProvider.tsx`) so toggling dark mode on a Quarto
  site carries over to StaffML and vice versa.

- **Playwright `site-audit.mjs`.** Single CLI with three subcommands:
  `sidebar` (asserts `#quarto-sidebar` / `.sidebar-navigation` is
  present and visible), `darkmode` (forces dark mode + scrolls + takes
  full-page screenshots into `_audit/`), `assets` (listens for failed
  network requests and navigation errors). Uses one runner for all
  three audits to keep dependencies minimal.

**Risk surface:** Visual changes only; no schema or workflow contracts
move. The dev-URL fix is a bugfix that already has a wrong-behavior
backstop (the old broken links 404 instead of taking users somewhere
unexpected).

---

## PR-3 — Scripts, audits, cleanup

**Commits:** `5be0675fb`, `1fd5746ab`, `a5df87562`, `27a7c2cb1`, `584458130`

Smaller-scoped polish + maintenance items:

- **Build-stamp.** New `shared/scripts/inject-build-stamp.sh` replaces a
  `<!-- MLSB_BUILD_STAMP -->` token in built HTML with
  `Last updated YYYY-MM-DD · <SiteLabel> · <CommitSHA>`. Token added to
  `book/quarto/config/shared/html/footer-common.yml` and
  `shared/config/footer-site.yml`. CSS hook (`.mlsb-build-stamp`) added
  to the shared head include. Wired into `kits-publish-live.yml` as a
  representative example; other publish-live workflows can adopt the
  step the same way.

- **PDF dropdown wiring.** `shared/config/navbar-common.yml` now
  surfaces TinyTorch Paper, MLSys·im Paper, and StaffML Paper in their
  respective dropdowns (file-pdf icon, opens in new tab). The paper
  build steps already produce these PDFs in CI; this just makes them
  discoverable.

- **Per-site 404 pages.** New `slides/404.qmd`, `instructors/404.qmd`,
  `site/404.qmd` so every subsite under mlsysbook.ai has a coherent
  404 instead of falling back to the GitHub Pages default white page.
  StaffML already has `not-found.tsx`. TinyTorch's legacy Sphinx 404
  is preserved (still wired on the Sphinx site that hasn't migrated).

- **Mirror-drift guard.** New `check-shared-mirrors` pre-commit hook
  runs `bash shared/scripts/sync-mirrors.sh --check` to fail the commit
  if any per-subsite real-file mirror has drifted from its canonical
  source in `shared/`. (Quarto's resource-copy step preserves symlinks
  instead of dereferencing them, so we need real copies.)

- **Duplication audit.** New `shared/scripts/find-duplicates.py` SHA-1
  hashes source-y files across ecosystem roots, subtracts the
  intentional groups from `sync-mirrors.sh`, and reports the rest as
  unintended duplicates (JSON to `.audit/duplicates.json`; `--strict`
  for non-zero exit). Initial cleanup: removed
  `tinytorch/scripts/cleanup_repo_history.sh` (byte-identical leftover;
  canonical lives at `tinytorch/tools/maintenance/cleanup_history.sh`).
  Remaining `runHistoryProvider.ts` triple in three vscode-ext packages
  is real refactor work — captured for later, out of scope here.

**Risk surface:** Build-stamp is opt-in per workflow (presence of token
required). Mirror-drift hook will fail commits that change a mirror
without re-syncing — that's the design, but heads-up to anyone editing
shared partials.

---

## PR-4 — TinyTorch release prep (LICENSE + v0.10.0 + workflow)

**Commits:** `493439b3d`, `8487536f7`, `bc3c80145`

Standalone TinyTorch maintenance work that lines up with the Vol II
launch:

- **MIT LICENSE.** `tinytorch/LICENSE` was a generic Apache-2.0 template
  stub with no copyright holder filled in, while `pyproject.toml`,
  `settings.ini`, and the README badge all declared MIT. Replaced with
  the standard MIT text attributed to "President and Fellows of Harvard
  College and the TinyTorch contributors". Appended a NOTICE block
  documenting the dual-license boundary: TinyTorch *software* (code,
  CLI, packaging) → MIT; TinyTorch *educational content* (chapter
  narratives, scaffolded notebooks) → CC-BY-NC-SA 4.0 (matches book/,
  labs/, kits/, slides/, instructors/, mlsysim/). For mixed files
  (notebooks with both code cells and prose), code cells fall under
  MIT and prose under CC-BY-NC-SA. **No permission expansion** — every
  file's metadata already said MIT; this just removes the paperwork
  inconsistency.

- **Version bump 0.1.9 → 0.10.0** in `pyproject.toml`, `settings.ini`,
  and the legacy Sphinx `announcement.json`. Verified
  `tinytorch.__version__` reads the new value. Added
  `tinytorch/CHANGELOG.md` documenting the v0.10.0 entry and the
  release-tag format (`tinytorch-vX.Y.Z`).

- **Workflow improvements.** `tinytorch-publish-live.yml` now accepts
  an `explicit_version` workflow_dispatch input that bypasses the
  patch/minor/major auto-bump (validated against
  `^[0-9]+\.[0-9]+\.[0-9]+$`, refuses to re-tag existing). Used for
  the 0.1.x → 0.10.0 jump. Also added `settings.ini` to the auto-bump
  step and the `git add` list so future releases keep nbdev metadata
  in sync (it was silently drifting before).

**Risk surface:** Only the LICENSE swap is permanent and visible.
Version files are consumed at import time — verified locally. Workflow
changes are additive (existing `release_type` path unchanged).

---

## PR-5 — Cutover skeletons (rollback / redirects / sitemap)

**Commits:** `30448b60e`, `d7e59fff5`, `bc35bcb68`

Plumbing that the actual cutover PRs will use; nothing here runs by
itself:

- **`shared/scripts/rollback-legacy.sh`.** Snapshot + restore for the
  gh-pages root, subsite-aware (never touches `book/`, `tinytorch/`,
  `kits/`, `labs/`, `mlsysim/`, `slides/`, `instructors/`,
  `interviews/`, `staffml/`, `about/`, `community/`, `newsletter/`).
  Three modes: `snapshot` (timestamps current root → `legacy-backup/<TS>/`),
  `restore <ID>` (copies a snapshot back to root), `list` (enumerates
  available snapshots). Dry-run by default; `--apply` to push.
  Operates against a fresh shallow clone in mktemp (never dirties the
  caller's working tree).

- **`shared/config/redirect-map.json` + `shared/scripts/build-redirects.py`.**
  Declarative legacy-URL → new-URL map. Generator emits two outputs from
  the same source of truth:
  - HTML stubs: `<meta refresh>` + `<link rel="canonical">` +
    `<meta robots="noindex,follow">` — the closest GitHub-Pages-friendly
    approximation of a 301. Real users redirect in <100ms; crawlers
    drop the legacy URL on recrawl and PageRank flows through the
    canonical.
  - Netlify `_redirects` file: same map, real 301s for the day we
    move off GitHub Pages.

  Map ships with seven seed entries demonstrating the patterns
  (single-page redirects + trailing-`*` wildcard subtree moves).
  Populating the full inventory from the legacy mlsysbook.ai sitemap
  is a separate task.

- **`shared/scripts/build-sitemap.py` +
  `.github/workflows/infra-build-sitemap.yml`.** Aggregator: walks the
  deployed gh-pages tree, finds every `<subsite>/sitemap.xml`, writes a
  single `sitemap-index.xml` at `<root>/sitemap.xml`, and adds the
  index to `<root>/robots.txt` via the standard `Sitemap:` directive.
  Skips `legacy-backup/`, `_archive/`, `_site/`. Optional
  `--include-subsite` allowlist for staged rollouts. Wrapped as a
  reusable workflow that joins the existing `gh-pages-deploy`
  concurrency group, so any `*-publish-live` workflow can call it as a
  final step without racing pushes.

**Risk surface:** Zero, because nothing here is wired into a publish
workflow yet. Each per-property launch PR will add its own call.

---

## What is intentionally NOT in this branch

A few items from the original release-prep planning conversation were
deliberately deferred so they can land at the right point in the rollout
rather than as untriggered plumbing now:

1. **Per-publish wiring of build-redirects + sitemap aggregation.** The
   skeletons in PR-5 should be invoked from each `*-publish-live.yml`
   as it ships its launch PR, not preemptively from this branch.

2. **Full legacy-URL inventory.** `redirect-map.json` ships with seed
   entries that demonstrate the patterns. Populating from the legacy
   mlsysbook.ai sitemap should happen against the actual legacy site
   the day before cutover, not against an estimate now.

3. **Volume I version bump (0.5.1 → 0.6.0).** Discussed but not landed
   here because the version bump should ride alongside the actual Vol I
   publish PR (cleaner blame, single source of "we shipped this").

4. **Volume II 0.1.0 baseline tag.** Same reasoning — should ride with
   the actual Vol II publish PR.

5. **`mlsysim` MIT relicensing.** Currently CC-BY-NC-SA, which is the
   wrong license class for what is fundamentally a software simulator.
   Out of scope for this PR (which was scoped to TinyTorch); flagged as
   a follow-up.

6. **TinyTorch legacy Sphinx site cleanup.** `tinytorch/site/` is mostly
   a dead Sphinx tree at this point; `site-quarto/` is the live source.
   Deleting the Sphinx tree is its own PR (touches `_static/`,
   `_config.yml`, `_toc*.yml`, etc.) and not blocking on launch.

7. **`runHistoryProvider.ts` deduplication across the three vscode-ext
   packages.** Real refactor work (needs a shared package); flagged in
   the PR-3 commit message.

---

## Verification done in this branch

- `python3 -c "import yaml; yaml.safe_load(open(F))"` on every
  modified/created workflow.
- `python3 .github/scripts/check_workflow_fork_safety.py` on every
  modified/created workflow (all pass).
- `bash -n` on every shell script.
- `python3 -c "import tinytorch; print(tinytorch.__version__)"` →
  `0.10.0` (confirms version file edits propagate correctly).
- `build-redirects.py --check` against the seed map (validates schema).
- `build-sitemap.py --check` against a synthetic tree (correctly skips
  `legacy-backup/`).
- `find-duplicates.py` against the worktree (correctly skips symlinks
  and `sync-mirrors`-declared intentional groups).

What was NOT verified locally:

- The reusable workflows themselves running on a real PR. They depend
  on `gh-pages` content and runner-resident clones; first real exercise
  will be when this branch is merged into `dev` and a `*-validate-dev`
  job picks them up.
- Browser-rendered visual changes (theme persistence, sidebar visibility
  in dark mode). The `site-audit.mjs` script exists for this but takes
  full-page screenshots rather than asserting pixel correctness — needs
  a human eyeball pass on the next dev-mirror deploy.

---

## Branch metadata

- Branch: `release-prep/staged-rollout-foundation`
- Worktree: `/Users/VJ/GitHub/MLSysBook-release-prep`
- Diverged from: `dev`
- Commits: 18, organized into five PR groupings
- Scope: `.github/workflows/` (new + modified), `shared/` (new files
  only), `tinytorch/` (LICENSE + version bump + CHANGELOG), per-subsite
  `404.qmd` for three subsites, three single-line config touches in
  Quarto configs (announcement bar split + footer token).
