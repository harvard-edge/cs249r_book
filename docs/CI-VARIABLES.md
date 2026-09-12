# CI variables

Repository-level variables that the GitHub Actions workflows read. They are
set under:

> Settings → Secrets and variables → Actions → Variables

A change takes effect on the next run of every workflow that reads the
variable. No code changes are required.

Variables hold values that are the same for every branch: tool versions, the
production domain, deploy paths on `gh-pages`, and service URLs. **Directory
paths inside the repository are not variables.** Workflows write them
literally for two reasons. The repository layout is versioned per branch while
a repository variable is shared by all branches, so a path stored in a
variable breaks whichever branch it does not match. And workflows triggered by
pull requests from forks receive no repository variables at all
(`.github/scripts/check_workflow_fork_safety.py` enforces this for
`pull_request` workflows).

Workflows that read a variable use `${{ vars.NAME || 'fallback' }}`, with the
fallback equal to the stored value, so a fork or a deleted variable still works.

## Deploy paths (gh-pages subpaths)

Where each project deploys on the `gh-pages` branch. The URL becomes
`https://<domain>/<deploy-path>/`.

| Variable | Value | Used by |
|---|---|---|
| `DEV_STAFFML_PATH` | `staffml` | staffml-publish-live, staffml-preview-dev |
| `DEV_TINYTORCH_PATH` | `tinytorch` | tinytorch workflows |
| `DEV_KITS_PATH` | `kits` | kits-publish-live |
| `DEV_LABS_PATH` | `labs` | labs-publish-live |
| `DEV_MLSYSIM_PATH` | `mlsysim` | mlsysim-publish-live |
| `DEV_INSTRUCTORS_PATH` | `instructors` | instructors-publish-live |
| `DEV_SLIDES_PATH` | `slides` | slides-publish-live |
| `VOL1_DEPLOY_PATH` | `vol1` | book-publish-live |
| `VOL2_DEPLOY_PATH` | `vol2` | book-publish-live |

## Cross-cutting versions and URLs

| Variable | Value | Used by | Why centralize |
|---|---|---|---|
| `NODE_VERSION` | `20` | 20+ workflows | Coordinated Node upgrades |
| `PYTHON_VERSION` | `3.12` | 20+ workflows | Coordinated Python upgrades. **Note:** mlsysim is pinned at `3.11` separately (intentional; it keeps the mlsysim hash stable). Don't change it without verifying the Merkle hash. |
| `PRODUCTION_DOMAIN` | `https://mlsysbook.ai` | Functional URLs in env vars and the canonical href in deployed HTML | Domain renames stay in one place |
| `STAFFML_VAULT_WORKER_URL` | `https://staffml-vault.mlsysbook-ai-account.workers.dev` | staffml-publish-live, staffml-preview-dev | Worker rename or migration |

## Other

| Variable | Value |
|---|---|
| `DEV_REPO` | `harvard-edge/cs249r_book_dev` |
| `DEV_REPO_URL` | `git@github.com:harvard-edge/cs249r_book_dev.git` |

## Retired: directory path variables

These variables named repository directories. Workflows on `dev` no longer
read them; each path is written in the workflow instead. `main`'s workflows
still read them, so leave them in place until the next `dev` → `main` publish
carries these workflows to `main`, then delete them.

For most of them the stored value already equals the path the workflows use.
**`BOOK_*`, `STAFFML_ROOT`, `VAULT_DIR`, and `VAULT_CLI_DIR` do not match, and
changing them before `main` has these workflows breaks `main`'s publishes.**

| Variable | Stored value | Path the workflows now use |
|---|---|---|
| `BOOK_ROOT` | `book` | `binder` |
| `BOOK_QUARTO` | `book/quarto` | `books` |
| `BOOK_TOOLS` | `book/tools` | `binder/tools` |
| `BOOK_DOCKER` | `book/docker` | `binder/docker` |
| `BOOK_DEPS` | `book/tools/dependencies` | `binder/tools/dependencies` |
| `STAFFML_ROOT` | `interviews/staffml` | `staffml/app` |
| `VAULT_DIR` | `interviews/vault` | `staffml/vault` |
| `VAULT_CLI_DIR` | `interviews/vault-cli` | `staffml/vault-cli` |
| `TINYTORCH_ROOT`, `TINYTORCH_SITE`, `TINYTORCH_SRC`, `TINYTORCH_TESTS` | `tinytorch`, `tinytorch/quarto`, `tinytorch/src`, `tinytorch/tests` | same |
| `MLSYSIM_ROOT`, `MLSYSIM_DOCS` | `mlsysim`, `mlsysim/docs` | same |
| `SLIDES_ROOT`, `INSTRUCTORS_ROOT` | `slides`, `instructors` | same |
| `KITS_ROOT`, `KITS_DOCS`, `LABS_ROOT`, `LABS_DOCS` | `kits`, `kits`, `labs`, `labs` | same |

Once `main` has these workflows, run the checked script. It deletes nothing
while `origin/main` or `origin/dev` still has a workflow that reads one of the
variables:

```bash
.github/scripts/retire-path-variables.sh          # check both branches, list what is still set
.github/scripts/retire-path-variables.sh --apply  # delete them
```

## Cloudflare zone id

`infra-cloudflare-purge` and `infra-cloudflare-redirects` read
`vars.CLOUDFLARE_ZONE_ID` and fall back to the `CLOUDFLARE_ZONE_ID` secret. The
purge workflow also looks the id up through the Cloudflare API when neither is
set. A zone id is not sensitive, so the variable is optional; its value is on
the `mlsysbook.ai` overview page in the Cloudflare dashboard.

## Secrets the workflows expect

Secrets hold credentials, so this list names them without values. Set one with
`gh secret set NAME -R harvard-edge/cs249r_book`, and list what is set with
`gh secret list -R harvard-edge/cs249r_book`.

| Secret | Used by |
|---|---|
| `SSH_DEPLOY_KEY` | every `*-preview-dev` workflow (deploys to `cs249r_book_dev`) |
| `BUTTONDOWN_API_KEY` | site-preview-dev, site-publish-live, site-refresh-stats, sync-newsletter |
| `GA4_SERVICE_ACCOUNT_JSON` | site-publish-live, site-refresh-stats |
| `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` | staffml-publish-live (vault worker deploy) |
| `CLOUDFLARE_CACHE_PURGE_TOKEN` | infra-cloudflare-purge |
| `CLOUDFLARE_REDIRECTS_TOKEN` | infra-cloudflare-redirects (Zone > Config Rules > Edit on `mlsysbook.ai`) |
| `CLOUDFLARE_ZONE_ID` | infra-cloudflare-purge, infra-cloudflare-redirects (fallback for the variable) |

## What is not a variable, and why

- **Directory paths**, for the reasons above. Moving a project means editing its
  workflows, which the `paths:` trigger filters already required.
- **`paths:` trigger filters** (the lists like `'staffml/app/**'` at the top of
  `*-validate-dev.yml`). GitHub Actions evaluates these when it loads the
  workflow, before variables resolve.
- **Comments and log strings** (for example `echo "Site: https://..."` in step
  summaries). Literal values are easier to grep and read in CI logs.
- **mlsysim's `python-version: '3.11'`**. The mlsysim Merkle hash is sensitive to
  the Python version, and a generic `vars.PYTHON_VERSION` bump must not silently
  invalidate hash equivalence.

## How to add a new variable

1. Confirm the value is the same on every branch and is duplicated across 3+
   workflow files. A repository path never qualifies.
2. Pick a name in `UPPER_SNAKE_CASE`, with a project prefix for project-specific
   values (`STAFFML_VAULT_WORKER_URL`, not `WORKER_URL`).
3. Set it:

   ```bash
   gh variable set MY_NEW_VAR -R harvard-edge/cs249r_book --body "the value"
   ```

4. Read it in workflows as `${{ vars.MY_NEW_VAR || 'the value' }}`. Workflows that
   run on `pull_request` cannot read variables; use a workflow-level `env:`
   constant there.
5. Add an entry to this document.

## Auditing

To see all variables currently set:

```bash
gh variable list -R harvard-edge/cs249r_book
```

To see where a variable is referenced:

```bash
git grep "vars\.NODE_VERSION" .github/
```
