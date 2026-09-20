# TinyTorch Changelog

All notable changes to the TinyTorch software package are documented here.
The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches `v1.0.0`. Until then, minor version bumps may include
backwards-incompatible changes that are noted explicitly under "Breaking".

Releases are tagged in git as `tinytorch-vX.Y.Z` (the package-prefixed tag
distinguishes TinyTorch releases from book volume releases that share this
monorepo). See [GitHub Releases](https://github.com/harvard-edge/cs249r_book/releases?q=tinytorch)
for the canonical list.

## [Unreleased]

### Added
- **Vector diagrams across all 20 modules**: ASCII pipeline art replaced with bespoke SVGs and markdown tables, with LaTeX rendered through MathJax, so notebook figures stay legible at any zoom and in print.
- **`tools/check_grading_holes.py`**: builds each module's real student-tier source, runs it with no student code, and reports which graded cells still pass. Ships with its own upper-bound caveat, because modules that import their own export target resolve it to whatever `tinytorch/` was last exported.
- **Release gate `grading: no graded cell scores only pre-solved scaffold code`**: catches a module that awards points while stripping nothing for the student. 34 fast gates.
- **`im2col` documented end to end**: Module 09 states the trade it can only describe, Module 17 builds and times the lowering, and book chapters 9 and 17 now hand off to each other explicitly.

### Changed
- **Student release no longer instructs students to implement solved code**: `apply_release_tier` drops the `TODO`/`APPROACH`/`EXAMPLE`/`HINT` briefing from a cell whose every region is kept pre-solved. 108 such cells across 19 modules shipped with a docstring reading "TODO: Implement ..." directly above the finished implementation; the briefing now survives exactly when the student still has work to do, or when the tier is the instructor reference.
- **Classroom-release target moved** from Fall 2026 to Spring 2027.
- **Hardware extensions** repaired and no longer stamped DO NOT EDIT; book chapter 21 rewritten around the extensions TinyTorch actually ships.
- **Module pages** given a uniform "What's next" shape and consistent prerequisites wording.

### Fixed
- **Six documented commands that exited with an argparse error**: `tito module start --no-browser` (real flag `--no-jupyter`), `tito milestone run --non-interactive` (`--skip-checks`), `tito setup -f` (`--force`), the nonexistent `tito system jupyter start/stop/status` subcommands, and `tito module status --student` / `--export` in `INSTRUCTOR.md` (the real surface is `tito nbgrader report`).
- **Core-region count**: stated as 47 in four places and 41 in a fifth; the measured value is 52 stripped of 241 total, with 189 shipping pre-solved.
- **Broken internal links** in the guide, and a notation table that rendered unprettified.
- **A paragraph rendering as a numbered section heading** in book chapter 7, caused by a `---` rule with no blank line above it, which Pandoc read as a setext H2 and printed in the table of contents as 7.3.
- **A false byte-level BPE claim**: a Sanity Check listed TinyGPT alongside GPT-2 and LLaMA as never emitting `<UNK>`, while `BPETokenizer` is character-level and the chapter's own trace loses three characters of "naïve" to `<UNK>`.
- **Numerical scales and proportions** calibrated across module charts and vector diagrams.
- **Milestone 01** labels its all-wrong runs honestly, and EOF no longer cancels dataset downloads silently.
- **Challenge tier** described accurately: no region carries `role="challenge"`, so the tier refuses to stage rather than producing "236 full blocks".

## [0.1.13] — 2026-09-16

### Added
- **Full End-to-End Pedagogical Certification**: Automated student journey simulation executing all 20 modules progressively from scratch, certified through 35 pre-release release gates.
- **Enhanced NBGrader 3-Tier Staging**: Student (52 core regions), Challenge (no annotated regions yet, so the tier refuses to stage), and Instructor (full solutions) release tiers with automated solution stripping and nbgrader schema validation.
- **Three-Phase Educational Test Runner**: `tito module test` with inline notebook assertions, educational pytest explanations, and cumulative cross-module integration tests.
- **Narrative Book Code Synchronization**: Real-time listing quotation from source modules into textbook chapters via `book/tools/listings.py`.
- **Hardware Extensions Track**: Optional hardware optimization modules (`tinytorch.extensions`) for AVX2/NEON SIMD, Apple Silicon Metal (MPS), and OpenAI Triton kernels.

### Changed
- Refined student notebook generation from `src/` to prevent solution leakage while retaining pedagogical scaffolding.
- Hardened `tito module start`, `view`, `complete`, and `reset` lifecycle workflows with atomic file staging and clear prerequisite feedback.

## [0.1.10] — 2026-04

The release that accompanied the Volume II launch. This is not a 1.0.
TinyTorch remains pedagogical-first and its APIs may still change between
modules.

### Added
- **Licensing clarity.** TinyTorch is now distributed under MIT (replaces
  the leftover Apache-2.0 stub `LICENSE` file). A NOTICE block at the
  bottom of `LICENSE` documents the MIT-vs-CC-BY-NC-SA boundary between
  TinyTorch *software* and the surrounding *educational content*.
- **Explicit-version release path.** `tinytorch-publish-live` workflow now
  accepts an `explicit_version` input that bypasses the major/minor/patch
  auto-bump for non-incremental jumps (used for this 0.1.x → 0.1.10
  release). Ordinary releases continue to use `release_type`.
- **`settings.ini` covered by automated bumps.** The publish workflow
  previously updated `pyproject.toml`, `install.sh`, the Quarto
  announcement, and the README badge but skipped `settings.ini`, which
  silently drifted. The workflow now keeps it in sync with each release.

### Changed
- Version bumped from `0.1.9`    → `0.1.10` in `pyproject.toml`,
  `settings.ini`, and the legacy site announcement banner.

### Notes for downstreams
- `tinytorch/__init__.py` reads its version from `pyproject.toml` at import
  time, so `import tinytorch; tinytorch.__version__` reflects the new
  number with no further changes.
- The MIT relicensing is a clarification, not a permission expansion: the
  pyproject metadata and README badge already declared MIT; only the
  `LICENSE` file text was wrong (it was an Apache-2.0 template stub with
  no copyright holder filled in, never actually granted to anyone).

## [0.1.9] and earlier

See [GitHub Releases](https://github.com/harvard-edge/cs249r_book/releases?q=tinytorch)
for pre-0.10 history. The 0.1.x line tracked early-access content
iteration during the Volume II writing process.
