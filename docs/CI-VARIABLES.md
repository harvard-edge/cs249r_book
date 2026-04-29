# CI variables

Repository-level variables that the GitHub Actions workflows read. Each
appears as `${{ vars.NAME || 'fallback' }}` in workflows — so the
workflows still run if a var is unset, but you can change behavior in
one place by editing the var's value at:

> Settings → Secrets and variables → Actions → Variables

If you change a var, the change takes effect on the next workflow run
of any workflow that reads it. No code changes are required.

## What's defined today

The values shown below are the **current** values stored in the repo.
The fallback in each workflow is hardcoded to match — if the var is
deleted, builds keep working until someone changes the underlying
default.

### Project source roots

These tell each project's workflows where its source lives.

| Variable | Current value | Used by |
|---|---|---|
| `STAFFML_ROOT` | `interviews/staffml` | staffml-publish-live, staffml-preview-dev, staffml-validate-dev |
| `VAULT_DIR` | `interviews/vault` | staffml-publish-live, staffml-preview-dev, staffml-validate-dev, staffml-validate-vault |
| `VAULT_CLI_DIR` | `interviews/vault-cli` | same as above |
| `TINYTORCH_ROOT` | `tinytorch` | (existing — pre-cutover) |
| `TINYTORCH_SITE` | `tinytorch/quarto` | tinytorch-publish-live |
| `BOOK_ROOT` | `book` | (existing — pre-cutover) |
| `BOOK_QUARTO` | `book/quarto` | (existing — pre-cutover) |
| `MLSYSIM_ROOT` | `mlsysim` | mlsysim-publish-live, mlsysim-preview-dev |
| `MLSYSIM_DOCS` | `mlsysim/docs` | same as above |
| `KITS_ROOT` | `kits` | kits-publish-live |
| `LABS_ROOT` | `labs` | labs-publish-live |
| `INSTRUCTORS_ROOT` | `instructors` | instructors-publish-live (also hardcoded as `instructors` in some places) |

### Deploy paths (gh-pages subpaths)

Where each project deploys on the `gh-pages` branch. URL becomes
`https://<domain>/<deploy-path>/`.

| Variable | Current value | Used by |
|---|---|---|
| `DEV_STAFFML_PATH` | `staffml` | staffml-publish-live, staffml-preview-dev |
| `DEV_TINYTORCH_PATH` | `tinytorch` | (existing) |
| `DEV_KITS_PATH` | `kits` | kits-publish-live |
| `DEV_LABS_PATH` | `labs` | labs-publish-live |
| `DEV_MLSYSIM_PATH` | `mlsysim` | mlsysim-publish-live |
| `DEV_INSTRUCTORS_PATH` | `instructors` | instructors-publish-live |
| `DEV_SLIDES_PATH` | `slides` | slides-publish-live |
| `VOL1_DEPLOY_PATH` | `vol1` | book-publish-live |
| `VOL2_DEPLOY_PATH` | `vol2` | book-publish-live |

### Cross-cutting versions and URLs

These cut across multiple projects and benefit most from
centralization.

| Variable | Current value | Used by | Why centralize |
|---|---|---|---|
| `NODE_VERSION` | `20` | 20+ workflows | Coordinated Node upgrades |
| `PYTHON_VERSION` | `3.12` | 20+ workflows | Coordinated Python upgrades. **Note:** mlsysim is pinned at `3.11` separately (intentional — keeps mlsysim hash-stable). Don't change without verifying the Merkle hash. |
| `PRODUCTION_DOMAIN` | `https://mlsysbook.ai` | Functional URLs in env vars + canonical href in deployed HTML | Domain renames stay in one place |
| `STAFFML_VAULT_WORKER_URL` | `https://staffml-vault.mlsysbook-ai-account.workers.dev` | staffml-publish-live, staffml-preview-dev | Worker rename / migration |

### Other (existing)

These were set before this convention landed. Documented here for
completeness.

| Variable | Current value |
|---|---|
| `BOOK_DEPS` | `book/tools/dependencies` |
| `BOOK_DOCKER` | `book/docker` |
| `BOOK_TOOLS` | `book/tools` |
| `KITS_DOCS` | `kits` |
| `LABS_DOCS` | `labs` |
| `SLIDES_ROOT` | `slides` |
| `TINYTORCH_SRC` | `tinytorch/src` |
| `TINYTORCH_TESTS` | `tinytorch/tests` |
| `DEV_REPO` | `harvard-edge/cs249r_book_dev` |
| `DEV_REPO_URL` | `git@github.com:harvard-edge/cs249r_book_dev.git` |

## What's NOT vars-ified (and why)

- **`paths:` trigger filters** in workflows (the lists like
  `'interviews/staffml/**'` at the top of `*-validate-dev.yml`).
  GitHub Actions evaluates these at workflow-load time, *before* vars
  are resolved. If you rename a project root, update the trigger
  filters by hand. The trigger filter is the only thing left in those
  files that needs manual editing.
- **Comments and log strings** (e.g. `echo "Site: https://..."` in
  step summaries). Literal values are easier to grep and read in CI
  logs. The functional places that ship to production HTML are
  vars-ified.
- **Inline Python heredocs** in `run:` blocks. Most reference paths
  via shell env vars (`os.environ["STAFFML_ROOT"]`) where the env
  flows from the job's `env:` block. A few short ones keep literal
  paths for readability — those are fine because they only run
  inside the workflow runner.
- **mlsysim's `python-version: '3.11'`** — left hardcoded because
  mlsysim's Merkle hash output is sensitive to the Python version,
  and we don't want a generic `vars.PYTHON_VERSION` bump to silently
  invalidate hash equivalence.
- **`interviews/staffml-vault-worker/`, `interviews/paper/`** — these
  are sibling top-level projects under `interviews/`, not subpaths
  of `STAFFML_ROOT`. Renaming `STAFFML_ROOT` should NOT cascade to
  them, so they stay literal.

## How to add a new var

1. Identify a value duplicated across 3+ workflow files.
2. Pick a name in `UPPER_SNAKE_CASE`. Use a project prefix for
   project-specific values (`STAFFML_ROOT`, not `ROOT`).
3. Set it via `gh variable set`:

   ```bash
   gh variable set MY_NEW_VAR -R harvard-edge/cs249r_book \
     --body "the/value"
   ```

4. Update the workflows to use `${{ vars.MY_NEW_VAR || 'the/value' }}`.
   Keep the fallback identical to the current var value — that way the
   workflow keeps working if the var is ever deleted.
5. Add an entry to this document.

## How to change a value

If you want to rename something (e.g. move `interviews/staffml` to a
new location):

1. Rename the directory in the source tree.
2. Update the var in repo Settings → Variables (or via
   `gh variable set NAME --body "newvalue"`).
3. Update each workflow's **fallback** to match (the `'fallback'`
   string in `${{ vars.X || 'fallback' }}`) — this keeps forks and
   first-time clones working.
4. Update any `paths:` trigger filters by hand (these can't read vars).
5. Update this document.

The first three are the bulk; the var change alone is enough to make
all CI runs succeed once the directory moves.

## Auditing

To see all vars currently set:

```bash
gh variable list -R harvard-edge/cs249r_book
```

To see where a var is referenced:

```bash
git grep "vars\.STAFFML_ROOT" .github/
```
