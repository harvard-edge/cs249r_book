# Contributing to the ML Systems Textbook

The Machine Learning Systems project welcomes contributions from everyone. This project is maintained by a community of contributors from around the world. We appreciate your help!

Your contributions are welcome and can encompass a variety of tasks, such as:

- Identifying and reporting bugs or errors in the text
- Correcting typographical errors
- Improving chapter content or explanations
- Creating or improving figures and diagrams
- Adding quantitative examples or exercises
- Enhancing the accessibility of the material
- Suggesting topics for new content

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

If you are unsure about whether a contribution is appropriate, feel free to open an [issue](https://github.com/harvard-edge/cs249r_book/issues) or start a [discussion](https://github.com/harvard-edge/cs249r_book/discussions).

## Repository Structure

This is a **two-volume textbook** built with [Quarto](https://quarto.org/docs/get-started/):

- **Volume I: Introduction to Machine Learning Systems** — Foundations for single-machine ML systems
- **Volume II: Machine Learning Systems at Scale** — Distributed systems at production scale

The key directories are:

```
book/quarto/contents/
├── vol1/          # Volume I chapters (introduction, training, hw_acceleration, etc.)
├── vol2/          # Volume II chapters (distributed_training, inference, etc.)
├── core/          # Shared content (dl_primer, frameworks)
├── frontmatter/   # Preface, acknowledgments
└── backmatter/    # Appendices, bibliography
```

Each chapter directory contains:
- A `.qmd` file (the chapter source)
- An `images/` folder with `png/` and `svg/` subdirectories

Quarto configuration files are in `book/quarto/config/` with volume-specific variants (e.g., `_quarto-html-vol1.yml`, `_quarto-pdf-vol1.yml`).

## How to Contribute

### 1. Open an Issue

If there is an open issue for the contribution you would like to make, please comment on the issue to let us know you are working on it. If there is no open issue, please open one first.

### 2. Fork and Clone

Fork the repository on GitHub and clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/cs249r_book.git
cd cs249r_book
git remote add upstream https://github.com/harvard-edge/cs249r_book.git
```

The upstream remote is read-only. You will push to your fork and open a Pull Request to merge upstream.

### 3. Create a Branch

Always branch from `dev`, not `main`. Use descriptive branch names with the issue number:

```bash
git checkout dev
git pull origin dev
git checkout -b iss14-fix-typo-in-introduction
```

Examples: `iss5-add-new-example`, `iss42-fix-figure-caption`, `iss100-improve-training-section`.

### 4. Make Your Changes

Please make sure that your changes are consistent with the style of the existing content.

- **Chapter content** lives in `book/quarto/contents/vol1/` or `vol2/`. Each chapter has its own directory.
- **Images** go in the chapter's `images/png/` (raster) or `images/svg/` (vector) subdirectory.
- **Editorial standards**: For prose contributions, please review the style conventions in the repository. We follow an academic textbook register (active voice, quantitative claims, no blog-post informality).

### 5. Commit Your Changes

Stage files explicitly (do not use `git add .`):

```bash
git add book/quarto/contents/vol1/introduction/introduction.qmd
git add book/quarto/contents/vol1/introduction/images/svg/new-figure.svg
git commit -m "Fix caption formatting in introduction chapter (issue #14)"
```

### 6. Render the Book

Please render the book to verify your contribution does not raise errors or warnings:

```bash
# Render a specific volume (recommended for faster builds)
quarto render --profile vol1-html
quarto render --profile vol2-html
```

### 7. Push and Open a Pull Request

```bash
git push origin your-branch-name
```

**Submit PRs to the `dev` branch, not `main`.**

Open a Pull Request with a brief description and the issue number (e.g., "Fix typo in introduction (issue #14)").

Opening an early PR is encouraged. This will allow us to provide feedback on your contribution and help you improve it. GitHub Actions will run on your PR and generate the book, so you can verify that your contribution renders correctly.

- If your PR is a work in progress, please add `[WIP]` to the title.

## Bug Reports

When reporting errors or issues, please include:

1. **Which volume and chapter** is affected (e.g., "Vol1, Chapter: Training")
2. **Section or page** where the issue appears
3. **Description** of the error (typo, incorrect figure, broken cross-reference, etc.)
4. **Suggested fix** if you have one

## Contributor Recognition

We use [All Contributors](https://allcontributors.org) to recognize everyone who helps improve the book.

### How to Recognize a Contributor

After merging a PR or resolving an issue, comment:

```
@all-contributors please add @username for TYPE
```

### Contribution Types

| Type | Emoji | Use For |
|------|-------|---------|
| `doc` | 📖 | Wrote or improved content |
| `review` | 👀 | Reviewed chapters or PRs |
| `translation` | 🌍 | Translated content |
| `design` | 🎨 | Created diagrams or figures |
| `bug` | 🐛 | Found errors or typos |
| `ideas` | 💡 | Suggested improvements |

### Example

```
@all-contributors please add @contributor for doc, review
```
