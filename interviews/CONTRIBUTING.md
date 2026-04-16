# Contributing to StaffML

Thanks for your interest. This guide covers contributing to the StaffML vault
(question corpus) and the site (`interviews/staffml/`).

For the full architecture of the vault pipeline, see
[`vault/ARCHITECTURE.md`](vault/ARCHITECTURE.md). For the review ledger behind
it, see [`vault/REVIEWS.md`](vault/REVIEWS.md).

---

## Quick start — from clone to first-question-visible

```bash
# 1. Clone and pick the staffml worktree
git clone https://github.com/harvard-edge/cs249r_book.git
cd cs249r_book/interviews

# 2. Install vault-cli (Python 3.12+)
pip install -e vault-cli/[dev]
vault --version
pytest vault-cli/tests/

# 3. Explore the corpus
vault doctor
vault stats

# 4. Run the local API shim (Phase 3+; stub in Phase 0–1)
#    Serves the Worker endpoint surface from a local vault.db so you don't
#    need a Cloudflare account to develop the site.
vault api --port 8002 &

# 5. Run the site against your local API
cd staffml/
cp .env.example .env.local
# edit .env.local: NEXT_PUBLIC_VAULT_API=http://localhost:8002
pnpm install
pnpm dev
# visit http://localhost:3000
```

The goal is clone → question-visible in under 10 minutes on a fresh machine. If
it's longer, file an issue titled "CONTRIBUTING.md getting-started friction".

---

## What can I contribute?

| Contribution type | Where | How |
|---|---|---|
| New question | `vault/questions/<track>/<level>/<zone>/` | `vault new` |
| Fix a question | same | `vault edit <id>` |
| Reclassify a question | same | `vault move <id> --to <track>/<level>/<zone>` |
| New topic | `vault/taxonomy.yaml` | PR with a §7 entry in EVOLUTION.md |
| Website UX | `interviews/staffml/src/` | Next.js; see AGENTS.md in staffml if present |
| Worker API | `interviews/staffml-vault-worker/` (Phase 3+) | Wrangler project |
| Schema evolution | `vault/schema/` | RFC-style PR per EVOLUTION.md |
| New `vault-cli` subcommand | `vault-cli/src/vault_cli/commands/` | Land with tests + docs update |

---

## Workflow

### Branching

- Start from `dev` for standalone work: `git checkout -b feat/short-description dev`.
- One logical change per branch. Atomic commits. No `git add -A` on vault changes.
- No `Co-Authored-By` tags, no "made with <tool>" footers. Commit messages read like regular engineering work.

### Before opening a PR

```bash
vault check --strict                 # invariants
pytest vault-cli/tests/              # unit + integration + contract
vault codegen --check                # LinkML ↔ Pydantic/DDL/TS drift check (Phase 1+)
```

CI re-runs these. PRs are merge-blocked on red CI.

### PR review

- Corpus PRs: at least one maintainer review, CI green.
- Code PRs (vault-cli, worker): one review, CI green.
- Schema-evolution PRs: two reviews required once external-contributor onboarding opens (Phase 7+).
- Schema-breaking PRs must include a migration script under `vault-cli/migrations/`.

### Provenance honesty

Every question's `provenance` field must honestly reflect how it was made:

- `human` — written from scratch by a human.
- `llm-draft` — produced by `vault generate`; not yet human-reviewed.
- `llm-then-human-edited` — an LLM draft substantially revised by a human (the
  common case). `generation_meta.human_reviewed_at` records when.
- `imported` — from an external source (e.g., book, published paper). Include
  source in `tags`.

Misattributing LLM content as `human` is a correctness bug, not a style nit.

### Author attribution

`vault new` populates `authors` from your `git config user.email` via
`vault/contributors.yaml` (mapping from email → handle). Submit a PR updating
that file to add yourself.

For external PRs: commit signatures (GPG/SSH) or GitHub-verified-email match is
required — CI rejects `authors:` claims that don't match the committer
identity.

---

## Style

- Keep questions focused on a **single concept**. "Good napkin math" is usually
  a signal; "grab-bag of facts" is usually a code smell.
- Use realistic hardware specs — check `mlsysbook/constants.py` and the
  `vault/schema/models.yaml` registry for the canonical values.
- Paper-cite URL format only: `https://mlsysbook.ai/book/chapters/<slug>`.
- Scenarios are plaintext; solutions and napkin math can use restricted
  Markdown + KaTeX.

---

## Things that block external PRs from merging

1. **Provenance lie** — `vault mark-exemplar` or `vault promote --reviewed-by`
   fields that don't match git committer.
2. **Registry mutation** — any commit that deletes lines from
   `id-registry.yaml`. The registry is append-only.
3. **Schema mixing** — questions at different `schema_version` in the same PR.
4. **Unsigned schema-evolution PR** — schema bumps require two maintainer
   approvals.

---

## Phase-by-phase scope

As of Phase 0, the vault pipeline is scaffolded but not operational end-to-end.
What works today:

- `vault --version`, `vault --help`.
- `pip install -e vault-cli/[dev]` and `pytest`.
- Documentation (ARCHITECTURE.md, REVIEWS.md, TESTING.md, EVOLUTION.md, this file).

What's coming per Phase (see [`vault/ARCHITECTURE.md`](vault/ARCHITECTURE.md) §14):

- **Phase 1**: `new`, `edit`, `move`, `rm`, `restore`, `build`, `check`, `serve`, `api`. YAML split lands.
- **Phase 2**: `publish` + primitives, paper-exporter rewrite, rollback-symmetry CI.
- **Phase 3**: D1 + Worker + `@staffml/vault-types` + FTS5 load-test gate.
- **Phase 4**: Website cutover + service worker + rollback drill.
- **Phase 5**: Chain pre-reveal indicator + instrumentation.
- **Phase 6**: About-page paper prominence.

External contributions to `vault/questions/` become feasible at Phase 1 exit.

---

## Asking for help

- Architecture questions → read [`vault/ARCHITECTURE.md`](vault/ARCHITECTURE.md).
- "Why was X decided this way?" → check [`vault/REVIEWS.md`](vault/REVIEWS.md).
  Most non-obvious decisions map to a reviewer finding.
- "Is this bug or intended?" → open an issue with the command you ran and the
  output.

---

**Thanks for contributing.**
