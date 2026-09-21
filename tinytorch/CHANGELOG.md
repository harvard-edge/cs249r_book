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

## [0.1.14] — 2026-09-21

Patch release focused on **curriculum harmonization, milestone alignment, and platform robustness**. It delivers a comprehensive visual and pedagogical overhaul of the textbook (Swiss Modernist cover, 4-beat PyTorch chapter deconstructions, 100% full-width diagrams), formally aligns Milestone 05 (`transformer`) and Milestone 06 (`mlperf`), hardens the `tito` CLI across Windows and Linux, and resolves key bugs in KV-cache generation and autograd. Version bumped from `0.1.13` to `0.1.14`.

### 📖 Textbook & Curriculum Architecture
- **4-Beat Chapter Openings**: Standardized the opening sections of all 24 chapters to the structured 4-beat PyTorch deconstruction pattern (*Standard PyTorch*, *The Illusion*, *The Engineering Reality*, and *The Contract*).
- **Swiss Modernist Cover**: Redesigned the textbook cover in Swiss Modernist palette with centered TinyTorch wordmark, monospace inline code tagline, and the 20-module runtime bus terminating in TinyGPT.
- **Diagram Harmonization**: Standardized all main-column diagrams across the textbook to 100% full text width with aligned margins and crisp vector typography.
- **Streamlined Chapter Content**: Removed end-of-chapter exercises across all 24 chapters to focus student progression entirely on hands-on notebook implementations.
- **Vector diagrams across all 20 modules**: Replaced ASCII pipeline art with bespoke SVGs and markdown tables, with LaTeX rendered through MathJax for crisp display at any zoom and in print.
- **`im2col` Lowering Documented**: Connected Module 09 and Module 17 with an explicit cross-chapter hand-off detailing the memory-compute trade-off.

### 🏆 Milestones & Historical Progression
- **Milestone 05 Formalized as `transformer`**: Anchored to Vaswani et al. (2017) to prove multi-head attention routing on algorithmic sequence reversal and copying (`PYTHON` → `NOHTYP`).
- **Milestone 06 Formalized as `mlperf`**: Anchored to the MLPerf benchmark discipline (2018), taking `DigitMLP` through the Optimization Olympics (profiling, INT8 quantization, weight pruning, and BLAS acceleration) to certify the Pareto frontier.
- **Elevated TinyGPT Culmination**: Unified textbook narrative and roadmap SVGs to position TinyGPT as the overarching Generative AI capstone.

### 🔧 Tito CLI & Platform Robustness
- **Windows Process & Encoding Robustness**: Fixed subprocess crashes on Windows systems with cp1252 codepages by pinning UTF-8 encoding across all CLI file I/O (#1966, #1971) and hardened the installer against silent hangs (#1969).
- **Subcommand Fixes**: Repaired multiple CLI commands that previously threw argparse or runtime errors:
  - `tito setup` now honors `TINYTORCH_NON_INTERACTIVE` and survives corrupted `profile.json` files (#2013).
  - `tito module complete` now verifies integration tests across all modules rather than skipping silently (#2008).
  - `tito module start` and `view` validate notebook presence before reporting success (#2010).
  - `tito benchmark baseline` handles missing input gracefully (#2014).
  - `tito milestone info` and `status` cleanly format milestones without duplicate years (#2009, #2019).
  - Registered real login command `tito community login` (#2012).
- **nbdev Pinning**: Added `[tool.nbdev]` configuration to `pyproject.toml` and pinned `nbdev < 3.0.16` for reliable export builds.

### 🐛 Framework Correctness & Ops
- **KV-Cache Self-Attention Fix** (M18 `memoization`, #1953): Ensured cached generation step incorporates the current token within the self-attention window.
- **Convolution FLOP Counting** (M14 `profiling`): Corrected FLOP calculation formulas for `Conv2d` in the runtime profiler.
- **Atomic Checkpoints** (M08 `training`): Made `Trainer.save_checkpoint` an atomic write to prevent corrupted model states during interrupted training runs.
- **Gradient Preservation** (M07 `optimizers`): Prevented `Optimizer.__init__` from inadvertently discarding pre-existing parameter gradients.
- **DataLoader Indexing** (M05 `dataloader`): Added negative index support in `TensorDataset.__getitem__`.
- **Autograd Layer Cleanup** (M06 `autograd`): Removed dead reimplementations in the autograd graph and fixed broadcast-gradient tests.

### 🛡️ Grading & Certification Pipeline
- **34 Pre-Release Verification Gates**: Added automated release gates ensuring no graded cell awards points exclusively for pre-solved scaffold code.
- **`tools/check_grading_holes.py`**: Automated validation tool that runs student-tier sources without student code to detect zero-work grading holes.
- **Scaffold Briefing Hygiene**: `apply_release_tier` now strips `TODO`/`APPROACH`/`HINT` briefings from cells that ship pre-solved, eliminating confusing instructions on completed code.
- **Classroom Target**: Aligned official classroom-release target for Spring 2027.

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
