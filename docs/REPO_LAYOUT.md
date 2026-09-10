# Repository layout

The short version: **book sources live in `books/`. Everything that builds them
lives in `publishing/`. Build output is disposable and lands in `books/_build/`.**

```
books/                    ← all book sources; this is the Quarto project root
  vol1/  vol2/  vol3/  vol4/    one directory per volume, chapter per subdirectory
  shared/                       anything used by more than one volume
    frontmatter/                shared front matter
    backmatter/                 shared back matter
    _partials/                  Quarto includes; "_" keeps them out of scans
    assets/                     images, covers, fonts, CSL
    tex/                        LaTeX preamble and per-volume theme colours
    filters/                    Lua filters
    scripts/                    build-support scripts
    audits/  publish/  calc/
  config/                       _quarto-{html,pdf,epub}-vol{1..4}.yml
  _extensions/                  Quarto extensions (must sit at the project root)
  index-vol1.qmd … index-vol4.qmd    per-volume landing pages
  references.bib  references-vol3.bib  references-vol4.bib
  _variables.yml  404.qmd  .quartoignore
  _build/                       render output; gitignored, safe to delete

publishing/               ← the toolchain, not the book
  binder                        the CLI entry point (invoke it as ./binder)
  cli/                          binder implementation
  tools/                        audits, scripts, generators
  tests/  docker/  docs/  config/  vscode-ext/  .layout/

binder                    ← symlink to publishing/binder; run ./binder from the root
```

## Why it looks like this

Until 2026-09 the sources lived at `publishing/quarto/contents/vol{1..4}/`, and a
second tracked copy sat at `books/vol{1..4}/`. Nothing read the second copy, and
it had drifted: vol3 had 3 chapters there against 30 in the live tree. Editorial
work kept landing in the copy that fed nothing, where no render and no check
could see it. `docs/BOOKS_MIRROR_RETIREMENT.md` records what was in it and how
to recover anything from history.

There was also a `book` symlink pointing at `publishing/`, so `book/quarto/` and
`publishing/quarto/` were the same directory under two names, sitting next to a
`books/` that was a third thing. That symlink is gone. The CLI is now reached as
`./binder` from the repository root.

## Rules of thumb

- **Editing a chapter?** It is under `books/vol<N>/<chapter>/<chapter>.qmd`.
  There is no other copy. If you find yourself editing something under
  `publishing/` that looks like prose, stop; you are in the wrong tree.
- **Adding a shared asset?** `books/shared/assets/`. Reference it from a config
  as `shared/assets/...`, which is relative to the Quarto project root.
- **Paths inside `books/config/*.yml`** are relative to `books/`, so a chapter is
  `vol1/introduction/introduction.qmd`, not `contents/vol1/...`.
- **Build output** goes to `books/_build/`. It is gitignored and disposable;
  deleting it costs a rebuild and nothing else.
- **Running a check or a build?** `./binder check figures`, `./binder build pdf
  --vol1`. Run it from the repository root.

## One consequence worth knowing

While the sources were at `publishing/quarto/contents/`, the checks defaulted
their scan root to the Quarto project directory rather than the content
directory, so `./binder check tables` with no `--path` silently scanned nothing
and always reported success. Now that the project root and the content root are
the same directory, the default scan works, and a bare `./binder check <group>`
reports real findings. Scoped runs with `--path` behave exactly as before.

## One thing this change cannot fix from inside the repository

The GitHub Actions workflows read repository **variables** that are configured
in GitHub settings, not in the repo:

| Variable | Was | Should now be |
|---|---|---|
| `BOOK_ROOT` | `publishing` | `publishing` (unchanged; it is the toolchain) |
| `BOOK_QUARTO` | `publishing/quarto` | `books` |
| `BOOK_TOOLS` | `publishing/tools` | `publishing/tools` (unchanged) |
| `BOOK_DEPS` | — | unchanged |

`BOOK_QUARTO` is the one that must change. Until it is updated under
Settings → Secrets and variables → Actions → Variables, any workflow step using
`${{ vars.BOOK_QUARTO }}` as a working directory will fail to find the book.
