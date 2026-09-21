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

Patch release focused on curriculum alignment, milestone updates, and platform reliability. It brings a visual refresh of the textbook and cover, standardizes diagram widths across all chapters, clarifies Milestone 05 (`transformer`) and Milestone 06 (`mlperf`), hardens the `tito` CLI across Windows and Linux, and fixes edge cases in KV-cache generation and autograd. Version bumped from `0.1.13` to `0.1.14`.

### 📖 Textbook & Design Updates
- **Chapter Introductions**: Standardized chapter openings with clear PyTorch baselines, systems-level context, and API specifications.
- **Cover Refresh**: Updated the book cover with centered typography, a clean code tagline, and the 20-module runtime architecture diagram.
- **Diagram Formatting**: Standardized all main-column diagrams to full text width with consistent margins and vector typography.
- **Streamlined Content**: Removed end-of-chapter exercises across all chapters to keep students focused on the executable notebooks.
- **Vector Diagrams**: Replaced older raster diagrams across all 20 modules with SVGs and markdown tables.
- **Convolution Details**: Expanded documentation on `im2col` across Chapters 9 and 17, detailing the memory versus compute trade-off.

### 🏆 Milestones
- **Milestone 05 (`transformer`)**: Anchored to Vaswani et al. (2017) to verify multi-head attention routing on sequence reversal and copying.
- **Milestone 06 (`mlperf`)**: Anchored to the MLPerf benchmark discipline (2018), taking `DigitMLP` through profiling, INT8 quantization, weight pruning, and acceleration to establish the Pareto frontier.
- **TinyGPT Roadmap**: Clarified the curriculum progression to position TinyGPT as the culminating generative AI project.

### 🔧 Tito CLI & Platform Reliability
- **Windows Encoding Fixes**: Resolved subprocess crashes on Windows systems with cp1252 codepages by enforcing UTF-8 across all CLI file operations (#1966, #1971), and resolved installer hangs (#1969).
- **Command Fixes**:
  - `tito setup` now supports non-interactive execution and recovers from corrupted profile configurations (#2013).
  - `tito module complete` now runs integration tests across all completed modules (#2008).
  - `tito module start` and `view` verify notebook presence before reporting success (#2010).
  - `tito benchmark baseline` handles missing input without crashing (#2014).
  - `tito milestone info` and `status` cleanly format milestone names without duplicate text (#2009, #2019).
  - Cleaned up community login command routing (#2012).
- **Build Configurations**: Configured nbdev settings in `pyproject.toml` and pinned compatible dependencies.

### 🐛 Framework Fixes
- **KV-Cache Self-Attention** (M18 `memoization`, #1953): Ensured cached generation correctly includes the current token within the self-attention window.
- **FLOP Counting** (M14 `profiling`): Corrected FLOP calculation formulas for 2D convolutions in the profiler.
- **Atomic Checkpoints** (M08 `training`): Ensured `Trainer.save_checkpoint` writes atomically to prevent file corruption during training interruptions.
- **Optimizer Gradients** (M07 `optimizers`): Prevented `Optimizer.__init__` from inadvertently clearing existing parameter gradients.
- **DataLoader Indexing** (M05 `dataloader`): Added negative index support in `TensorDataset.__getitem__`.
- **Autograd Cleanup** (M06 `autograd`): Cleaned up redundant code in the autograd layer and fixed broadcast-gradient tests.

### 🎓 Grading & Student Notebooks
- **Notebook Validation**: Added automated checks to verify that graded notebook cells require active student implementation rather than awarding credit for pre-solved code.
- **Scaffolding Cleanup**: Stripped unnecessary docstring briefings from cells where solutions are intentionally provided, avoiding conflicting instructions for students.
- **Classroom Schedule**: Aligned the official course timeline for Spring 2027.

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
