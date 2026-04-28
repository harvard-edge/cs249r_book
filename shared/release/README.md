# Shared release infrastructure

Single source of truth for "what release is this?" across every publishable
artifact in the MLSysBook monorepo. Mirrors and generalizes the StaffML
pattern landed in `feat/staffml-version`. See `docs/VERSIONING.md` for
contributor-facing usage; this file is for operators and downstream
projects that need to know the contract.

---

## What's here

| File | Purpose |
|---|---|
| `release-pill.html` | Footer snippet — small "v0.1.0 · Apr 26" identity pill |
| `release-card.html` | About-page snippet — fuller release-identity card with copyable hash |
| `README.md` | This file |

The Python helpers and JSON schema live at the repo root under
`scripts/version/` so they can be invoked from any workflow without
relative-path gymnastics:

| Path | Purpose |
|---|---|
| `scripts/version/release.py` | CLI: compute-id, compute-hash, emit-release, emit-manifest |
| `scripts/version/schema.json` | JSON Schema for `releases/<id>/release.json` |

The reusable GitHub Actions workflow lives where workflows live:

| Path | Purpose |
|---|---|
| `.github/workflows/_release-publish.yml` | `workflow_call` — orchestrates bump + tag + release notes |

## Contract

Every project that adopts the pattern produces TWO build-time artifacts
on every publish:

1. **`releases/<project>-<release_id>/release.json`** — the canonical,
   commit-ready release artifact. Validates against
   `scripts/version/schema.json`. Contains `release_id`, `release_hash`
   (full hex digest over input bytes), `git_sha`, `created_at`,
   `input_paths`, and a `metadata` object with project-specific stats.
   Tier A also includes a `files: [{path, hash}, ...]` array (Merkle-ish
   per-file hashes) for partial verification.

2. **`<deployable>/release-manifest.json`** — the build-time projection
   the deployable bundles. Strict subset: `releaseId`, `releaseHash`,
   `schemaVersion`, `tier`, `project`, `buildDate`, plus a `metadata`
   object. Static sites deploy this at the site root; the footer pill
   fetches it via `<meta name="release-manifest" content="...">`.

A project's site may extend (1) and (2) with project-specific keys
(StaffML's vault-manifest.json adds `questionCount`, `trackDistribution`,
etc.) — but those keys live in `metadata`, never at the top level.

## Tiers

- **Tier A** (citable): full Merkle-style file index in `release.json`.
  Use for academically-cited content (paper hashes, textbook releases,
  StaffML question bank, TinyTorch framework releases).
- **Tier B** (lite): single SHA-256 over content directory; no per-file
  index. Use for rapidly-iterating content where citation isn't a
  primary concern (Kits, Labs, Instructors).

## Footer pill setup (Quarto)

Each Quarto project does ~3 lines of config. Example for a project
deployed at `https://mlsysbook.ai/<project-base>/`:

```yaml
# _quarto.yml
project:
  resources:
    - "../shared/release/release-pill.html"

format:
  html:
    include-in-header:
      - text: |
          <meta name="release-manifest" content="/<project-base>/release-manifest.json">
    include-after-body:
      - file: "../shared/release/release-pill.html"
```

The publish workflow drops `release-manifest.json` at the site root
(`<deploy_path>/release-manifest.json`) so the meta-tag URL resolves.

## Footer pill setup (Next.js / hand-rolled)

The pill is a React-free static snippet — works in any HTML. Set the
meta tag once in your layout, then drop the snippet wherever you want
the pill (typically the footer). See StaffML's `Footer.tsx` for an
inline-React equivalent that bakes the manifest at build time instead
of fetching at runtime — that approach is preferred for citation-
critical content and is what StaffML uses.

## Reusable workflow setup

Each project's `<project>-publish-live.yml` becomes a thin wrapper
calling `_release-publish.yml`. See StaffML's workflow for a full
example. The reusable workflow handles:

- Validates `confirm: PUBLISH` safety gate
- Computes `new_release_id` from `release_type` + previous tag
- Calls the project's build with the computed `release_id`
- Validates the manifest the build emitted (must agree with computed id)
- Tags `<project>-v<release_id>`
- Generates GitHub Release notes (AI-enhanced if `ai_release_notes=yes`)
- Uploads the deployable artifact

The project-specific build commands are passed in via `with:`. The
reusable workflow never assumes a particular build tool.

## When NOT to use this

- One-shot scripts or internal tools that don't get cited or deployed.
- Documentation that lives inside another project's repo (use the
  outer project's release identity).
- Anything that doesn't run through a publish workflow at all
  (Periodic-Table, currently — needs a publish workflow first).

## Validating an existing release

```bash
# From repo root, validate a release.json against the schema:
python3 -c "
import json, jsonschema
schema = json.load(open('scripts/version/schema.json'))
release = json.load(open('releases/staffml-0.1.0/release.json'))
jsonschema.validate(release, schema)
print('OK')
"
```

(`jsonschema` package required; install via `pip install jsonschema`.)

## See also

- `docs/VERSIONING.md` — contributor-facing how-to
- `interviews/staffml/src/lib/stats.ts` — StaffML's reference reader
- `.github/workflows/_release-publish.yml` — reusable workflow source
