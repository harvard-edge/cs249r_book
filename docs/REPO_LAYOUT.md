# Repository layout

This repository holds the ML Systems textbook and everything built around it:
the book sources, the toolchain that builds and checks them, the companion
software, the teaching materials, the StaffML interview-prep product, and the
sites that publish all of it. Each area has one home, and no directory is
reached through a symlink.

## Map

Each project with its own site sits at the top level under the name of its
URL, so `mlsysbook.ai/tinytorch/` is `tinytorch/` and `mlsysbook.ai/slides/` is
`slides/`. The folders that support every project follow.

```
books/            textbook sources; the Quarto project root for all four volumes
binder/           book toolchain: the binder CLI, checks, audits, build scripts, tests

tinytorch/        TinyTorch build-your-own-framework course, site, and tito CLI
mlsysim/          MLSys·im analytical modeling library, docs site, paper, tutorial
mlperf-edu/       MLPerf EDU laptop-scale benchmark suite and site
labs/             Marimo co-labs (the mlsysbook_labs package) and the labs site
kits/             hardware kit labs (Arduino, Seeed, Raspberry Pi) and site
slides/           Beamer lecture decks, one per chapter, and the slides portal
instructors/      The Blueprint, the instructor site
staffml/          StaffML interview-prep product
  app/              Next.js web app
  vault/            question corpus (YAML), taxonomy, release artifacts
  vault-cli/        Python CLI that builds, checks, and releases the vault
  vault-worker/     Cloudflare Worker serving the vault API
  vault-types/      shared TypeScript types
  paper/            StaffML paper
site/             mlsysbook.ai landing site: home, about, community, newsletter
socratiq/         SocratiQ AI learning widget
design-grammar/   ML systems design grammar catalog

shared/           assets several sites use: brand styles, navbar and footer config,
                  redirects, cross-site scripts, the release pill
docs/             repository-level docs: this file, CI variables, versioning
scripts/          standalone utilities outside the binder CLI (versioning, cross-reference audits)
tools/            monorepo-level audits, release smoke tests, historical cleanup manifests
wheels/           prebuilt wheels the labs WASM smoke test installs
README/           translated READMEs
.github/          workflows, workflow scripts, issue templates, the dev landing page
```

Root files cover the repository as a whole: `README.md`, `CONTRIBUTING.md`,
`SECURITY.md`, the citation and license files, `pyproject.toml` (pytest,
codespell, and lint settings), and `.pre-commit-config.yaml`.

## books/

```
books/
  vol1/ … vol4/       one directory per volume and one subdirectory per chapter
                      (vol1/introduction/introduction.qmd), plus parts/
  shared/             material more than one volume uses
    frontmatter/  backmatter/  _partials/  assets/  tex/  filters/
    scripts/  publish/  audits/  calc/
  config/             _quarto-{html,pdf,epub}-vol{1..4}.yml and shared YAML fragments
  _extensions/        Quarto extensions (they must sit at the project root)
  index-vol1.qmd …    per-volume landing pages
  references*.bib     bibliographies
  _build/             render output; gitignored and disposable
```

Paths inside `books/config/*.yml` are relative to `books/`. Quarto reads only
`books/_quarto.yml`, so the binder copies the chosen config there before each
build and copies the volume's `index-volN.qmd` to `books/index.qmd`. Both copies
are gitignored, and the first line of `_quarto.yml` names the file it came from.

## binder/

```
binder/
  binder              CLI entry point; run ./binder/binder from the repository root
  cli/                commands/, checks/, core/, formats/, data/
  tools/              audit/, audits/, scripts/, figures/, dependencies/, git-hooks/, setup/
  tests/              toolchain tests
  docker/             Linux and Windows build containers
  docs/               toolchain documentation
  config/  .layout/  vscode-ext/  socratiQ/
  postBuild  requirements.txt
```

mybinder.org reads its configuration from a top-level `binder/` folder, so the
Launch Binder links for TinyTorch are configured by `binder/postBuild` and
`binder/requirements.txt`, next to the toolchain.

## Rules of thumb

- **Editing a chapter?** It is `books/vol<N>/<chapter>/<chapter>.qmd`. There is
  no other copy.
- **Adding a book asset?** `books/shared/assets/`, referenced from a config as
  `shared/assets/...`.
- **Changing a style, navbar, or redirect shared across sites?** `shared/`.
- **Running a check or a build?** `./binder/binder check refs` or
  `./binder/binder build pdf --vol1`, from the repository root.
- **Importing mlsysim from source?** Put the `mlsysim/` project folder on the Python path.
  The root pytest config, the book configs, and the binder CLI already do.
- **Writing a workflow?** Write directory paths literally. Repository variables
  are shared by every branch while the layout is versioned per branch, so a path
  kept in a variable breaks whichever branch it does not match
  (`docs/CI-VARIABLES.md`).

## How it got here

The September 2026 reorganization moved every area to the home above:

| Before | After |
|---|---|
| `publishing/quarto/contents/vol{1..4}/` | `books/vol{1..4}/` (the old `books/` copy was retired; see `docs/BOOKS_MIRROR_RETIREMENT.md`) |
| `publishing/`, reached as `book/` and `./binder` | `binder/`, run as `./binder/binder` |
| `.binder/` | `binder/postBuild`, `binder/requirements.txt` |
| `interviews/staffml`, `interviews/vault*`, `interviews/staffml-vault-{worker,types}` | `staffml/app`, `staffml/vault*`, `staffml/vault-{worker,types}` |
| `_quarto.yml` symlinks in kits, labs, and the MLSys·im docs | real `_quarto.yml` files |
| `staffml/vault/releases/latest` symlink | `staffml/vault/releases/latest.txt` |

TinyTorch, MLSys·im, MLPerf EDU, the slides, and the instructor site stayed at
the top level, so their GitHub paths and site URLs still match.

Published URLs did not change: mlsysbook.ai keeps `/vol1/`, `/tinytorch/`,
`/mlsysim/`, `/slides/`, `/instructors/`, and `/staffml/`, and `/interviews/`
redirects to `/staffml/`.

## One consequence worth knowing

While the sources were at `publishing/quarto/contents/`, the checks defaulted
their scan root to the Quarto project directory rather than the content
directory, so a check with no `--path` silently scanned nothing and always
reported success. Now that the project root and the content root are the same
directory, the default scan works, and a bare `./binder/binder check <group>`
reports real findings. Scoped runs with `--path` behave as before.
