# Contributing to MLSysBook

Thanks for your interest in MLSysBook! This repository is the home for the
**ML Systems textbook** plus a family of sibling projects — TinyTorch, Co-Labs,
Hardware Kits, MLSys·im, MLPerf EDU, and StaffML. Most contributions land in
exactly one of those projects, so this top-level guide just gets you to the
right place.

> [!IMPORTANT]
> Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.
> Security issues should follow [SECURITY.md](SECURITY.md), not the public
> issue tracker.

## Pick your project

| If you want to... | Project | Read this guide |
|---|---|---|
| Fix a typo, improve a chapter, add a figure | **Textbook** | [`book/docs/CONTRIBUTING.md`](book/docs/CONTRIBUTING.md) |
| Add or fix a TinyTorch module / test / milestone | **TinyTorch** | [`tinytorch/CONTRIBUTING.md`](tinytorch/CONTRIBUTING.md) |
| Improve a hardware lab or board recipe | **Hardware Kits** | [`kits/README.md`](kits/README.md) |
| Add or fix an interactive Co-Lab | **Labs** | [`labs/README.md`](labs/README.md) |
| Contribute an MLSys·im model, scenario, or scorecard | **MLSys·im** | [`mlsysim/docs/contributing.qmd`](mlsysim/docs/contributing.qmd) |
| Add a workload to the MLPerf EDU benchmark suite | **MLPerf EDU** | [`mlperf-edu/README.md`](mlperf-edu/README.md) |
| Author or fix a StaffML interview question | **StaffML** | [`interviews/CONTRIBUTING.md`](interviews/CONTRIBUTING.md) |
| Improve teaching materials, syllabi, or rubrics | **Instructors** | [`instructors/README.md`](instructors/README.md) |
| Update slides for a chapter | **Slides** | [`slides/README.md`](slides/README.md) |

Not sure which one applies? Open a
[Discussion](https://github.com/harvard-edge/cs249r_book/discussions) and we'll
help route it.

## Universal policies (apply to every project)

These conventions hold across the whole monorepo. The per-project guides
specialize them.

### 1. Branch from `dev`, not `main`

`main` tracks the published live site. All work merges to `dev` first and ships
to `main` on release.

```bash
git checkout dev
git pull origin dev
git checkout -b iss123-short-descriptive-slug
```

Branch names should reference the issue number when one exists
(`iss42-fix-figure-caption`, `feat/tinytorch-conv-module`).

### 2. Set up pre-commit hooks (one time per clone)

This repo runs ~60 pre-commit checks (BibTeX validation, figure-div syntax,
markdown link checks, EPUB hygiene, vault corpus-guard, and more). They catch
issues that would otherwise burn maintainer review cycles.

```bash
./book/binder setup
```

If `pre-commit` is missing, install it (`pip install pre-commit`) and re-run.

### 3. Stage files explicitly

Do **not** use `git add .` — it's easy to commit unrelated edits, secrets, or
build artifacts. Stage paths individually:

```bash
git add book/quarto/contents/vol1/introduction/introduction.qmd
git commit -m "Fix caption formatting in introduction (issue #14)"
```

### 4. Open a Pull Request to `dev`

* Reference the issue number (`Fixes #123` or `Related to #456`).
* Mark drafts with `[WIP]` in the title or use GitHub's "Draft PR" mode.
* Use the [PR template](.github/PULL_REQUEST_TEMPLATE.md) — it asks the
  questions reviewers will ask anyway.
* CI will render the affected project (book / tinytorch / staffml / etc.); fix
  any failures before requesting review.

### 5. Code of Conduct

By contributing you agree to abide by the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Report concerns to
`vj@eecs.harvard.edu` or `nkhoshnevis@g.harvard.edu`.

### 6. License of contributions

By submitting a PR you agree to license your contribution under the project's
[license](LICENSE.md): Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International for content, with code components dual-licensed under their
project-local terms (see each sub-project's `LICENSE` for specifics).

## Reporting bugs and asking questions

* **Found a real bug or specific issue?** Open an
  [issue](https://github.com/harvard-edge/cs249r_book/issues) using the
  template that fits (we have eight: book, TinyTorch bug, MLSys·im bug, new
  challenge, interview question, StaffML report/contribute, and more).
* **General question or design discussion?**
  [Discussions](https://github.com/harvard-edge/cs249r_book/discussions) is the
  better fit.
* **Security issue?** See [SECURITY.md](SECURITY.md) — please do **not** open a
  public issue.

## Contributor recognition

We use the [All Contributors](https://allcontributors.org) bot. After your PR
merges, a maintainer (or you, on your own PR) can comment:

```text
@all-contributors please add @your-username for doc, code, ideas
```

You'll be added to the project's recognition table in the README. See
[`book/docs/CONTRIBUTING.md`](book/docs/CONTRIBUTING.md#contribution-types) for
the full list of contribution types.

---

Thanks for helping make MLSysBook better. The community runs on people who
take the time to fix one typo, file one good bug, or write one careful PR.
