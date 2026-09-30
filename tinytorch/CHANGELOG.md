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

## [0.2.0] — 2026-09-29

**TinyTorch: From Tensors to Transformers.** The first minor release. The 20 modules now carry students from a strided tensor to a transformer that writes Python, and every milestone checks the student's own code against an independent reference before it counts as passed. This release adds the transformer-era milestones, a rebuilt lab guide and companion book, and removes every place we found where a test, a milestone, or CI reported success without checking anything.

### ✨ Highlights
- **Transformer-era milestones**: Milestone 05 trains TinyGPT on Shakespeare and proves attention on sequence routing, with optional TinyCopilot (code completion on the bundled TinyPy corpus, gated on Python that parses for function names the model never saw) and conversational Q&A with an overfitting diagnosis. Milestone 06, *MLPerf to Generative Serving (2018)*, compresses models across three benchmark divisions and measures the student's KV cache. Milestone 07 checks custom kernels against a reference.
- **Try it**: After TinyCopilot and the KV-cache part pass, a prompt lets students type their own code prefix or sentence and watch their model respond. It runs only after the verdict and never affects it.
- **Milestones that cannot be gamed**: Each required part is recorded separately, and a milestone completes only when all required parts pass. Losses are re-measured in NumPy, and causality, attention-content and position probes catch a model that reaches a low loss without working attention or positional encoding. A knockout study broke each student module in turn and confirmed the milestones that use it fail.
- **Lab guide and book**: A rebuilt website with real recordings of the installer, the CLI and the milestones, and the companion book *TinyTorch: From Tensors to Transformers*, both read line by line against the code for this release.
- **One-line install you can trust**: `install.sh` pins the release tag, and every demo on the site is a recording of the real installer and real runs.

The sections below list the changes in detail.

### 🔧 CI & Release
- **Publish preflight runs its tests**: Inside the reusable `workflow_call`, the validate workflow read the caller's event name, resolved an empty `test_type`, skipped every stage, and still reported green (a 2026-09-23 publish run bumped, deployed, and tagged with zero tests run). It now reads `inputs`, fails on an unknown or empty `test_type`, and the Summary job requires every requested stage to succeed.
- **Wider CI coverage**: E2E runs `quick`, `module_flow` and `milestone_flow` (17 → 28 tests), so the check that a failing notebook cannot complete a module now runs in CI. A missing test path or module test directory fails instead of passing.
- **Workflow inputs via `env:`**: `publish-live` and `validate-dev` no longer interpolate free-text inputs or `head_ref` into shell scripts. Windows jobs stay non-blocking, but the Summary lists any Windows failure.

### 🏆 Milestones
- **Per-part recording**: `tito milestone run NN --part N` runs and records only that part; running Part 1 alone no longer records the whole milestone.
- **Reference checks**: Losses in Milestones 02–05 are re-measured against NumPy after training, and the transformer milestones add causality, attention-content (uniform attention no longer passes Milestone 05 Parts 3–4) and position probes.
- **Try-it prompts** (Milestone 05 Part 3, Milestone 06 Part 2): see Highlights. Ctrl-C at the prompt keeps the pass; Ctrl-C during a run still stops it without recording.
- **Milestone 06.2**: Restored the missing-module message and added a regression test for it.
- **Milestone 07 (`kernels`)**: Each kernel check now reports whether the native path actually ran and gates on an `allclose` tolerance. A mismatch fails the run; a run where no native kernel executed is INCOMPLETE (exit 1) instead of `[SUCCESS]`.
- **TinyCopilot (Milestone 05)**: The syntax gate scores held-out prompts, exactly as generated.
- **TinyGPT chat (Milestone 05)**: Overfitting is diagnosed from the measured loss curve rather than asserted.
- **Optimization Olympics (Milestone 06)**: Pareto labels come only from the computed frontier, the takeaway is built from measured values (no hardcoded "4×" or "100% accuracy"), and untrained CNN/GPT divisions report agreement with FP32 rather than accuracy.
- **Accuracy gates**: Milestones 03 (MLP) and 04 (CNN, TinyDigits) exit 1 below 75% test accuracy instead of always exiting 0. TinyGPT Shakespeare reports loss and perplexity and no longer claims coherent output.
- **`tito benchmark`**: `capstone` no longer saves a fake `basic_score: 75` or placeholder scores (it exits 1 until implemented); `baseline` is a NumPy environment check with a real geometric mean and no score; the "submit to the community?" prompt that led to a stub is gone and results are stored locally.

### 🐛 Framework Fixes
- **Dropout in eval mode** (M03 `layers`): Layers carry `self.training` with `train()`/`eval()`, `Dropout.forward(training=None)` falls back to it, and `set_training_mode()` walks sub-layers. `Trainer.evaluate` uses it, so non-`Sequential` models no longer drop units during evaluation.
- **Target gradients** (M06 `autograd`): MSE and BCE backward return a gradient for the target when it requires grad, so a loss between two model outputs (e.g. distillation) no longer drops half its gradient.
- **float32 parameters stay float32** (M07/M08): Gradient clipping and `CosineSchedule` produce Python-float coefficients, and SGD/Adam/AdamW cast updates back to the parameter dtype, so NumPy 2 promotion no longer silently doubles parameter memory.
- **Layer validation**: `MaxPool2d` rejects padding larger than half the kernel; `Conv2d` accepts a tuple stride; `CharTokenizer` deduplicates its vocabulary.
- **LoRA** (`extensions/lora.py`): `LoRALinear.parameters()` returns the adapter matrices, so an optimizer actually trains them (it previously inherited an empty list). The extension template's accelerated and fallback paths now compute the same ReLU.

### 🔧 Tito CLI & Platform Reliability
- **Login callback**: Binds `127.0.0.1` (`0.0.0.0` only under WSL), carries a per-login `state` nonce, accepts one callback, and writes the credential file `0600` at creation.
- **Progress sync**: Lists exactly what is sent; automatic sync requires recorded consent (`tito community sync --enable-auto/--disable-auto`, `TITO_NO_SYNC=1`); milestone totals come from the registry (7, not 6).
- **Sync opt-out is quiet**: After `--disable-auto`, completions no longer prompt to sync; `--enable-auto` syncs without asking.
- **Other**: The update check always verifies TLS; `tito system health` exits 1 when it reports issues; an installed package reports its version via `importlib.metadata`; `tito module status` with no modules explains instead of dividing by zero.

### 🎓 Grading
- **Value-checking graded tests**: Six nbgrader-graded tests (80 points) in Modules 09, 11, 12, and 13 asserted only shapes and configuration. They now compare outputs against an independent NumPy reference (convolution, positional and sinusoidal encodings, scaled dot-product and multi-head attention, MLP, transformer block). Point values and cell metadata are unchanged.
- **Tests that could not fail**: A test importing a nonexistent `BatchNorm1d` under `except ImportError: pass`, `assert True` endings, and milestone smoke tests guarded by `if hasattr(...)` now make real checks; loose tolerances on exact math are tightened; new direct tests cover attention scaling and the SGD/Adam/AdamW update rules.

### 📚 Book & Website
- **Release read**: Every book chapter and website page was read against the code at release. Fixed claims the code doesn't support (coherent Shakespeare output, "100% valid Python", guaranteed Pareto optimality, leaderboard submission), code/prose mismatches (GELU form, attention mask semantics, P90 gating, nonexistent APIs in snippets, which are now executed against the package), an invented script excerpt and an unsourced footnote, and arithmetic slips in worked examples and tables.
- **Module Dependencies**: Modules 05–20 follow the four-label spec again, and each section's lists now match the module's actual imports; `release_check --fast` passes all 34 gates.
- **Figures**: Diagrams that contradicted the code (additive mask, CSR break-even, Pareto "peak" labels, capstone stack) are corrected, and the SVG generators reproduce the corrected versions.

## [0.1.19] — 2026-09-23

Patch release fixing sequence-model evaluation, compile tracing, and a CLI import-order bug.

### 🐛 Framework Fixes
- **Sequence evaluation** (M08 `training`): `Trainer.evaluate` reduces along `axis=-1`, so predictions on `(batch, seq, vocab)` logits pick the vocabulary class.
- **Scalar tracing** (extensions `compile`): `TracedNode` handles scalar operands in `__add__`, `__radd__`, `__mul__`, and `__rmul__`.
- **Benchmark CLI**: `tito benchmark` imports `BaseCommand` and `TinyTorchCLIError` before the lazy RNG helper, fixing an import error.
- Added tests for 3D evaluation, scalar tracing, RNG import errors, and `LogSoftmax`; synced the `evaluate` book listing.

## [0.1.18] — 2026-09-22

Adds the Systems Extensions chapter and a round of book, cover, and Windows encoding fixes.

### 🧩 Extensions
- **Extending TinyTorch chapter** (`21_extensions`): One chapter covering ecosystem extensions, with working `tinytorch.extensions` implementations of LoRA, a loss scaler, checkpointing, and operator fusion, plus a pytest suite in `tests/extensions/`.

### 📖 Textbook & Design Updates
- Milestone introductions restructured around history and the breakthrough each reproduces; descriptions aligned with the `src/` modules.
- Matplotlib quantitative plots added across chapters; broken image links removed.
- Cover shows the 20-module runtime with expansion ports into the Capstone; subtitle updated to match.
- Standardized chapter subtitles, margin alignment, and Further Reading citation format; added MLPerf Tiny and foundational systems papers.
- Preface motivates building ML systems from scratch alongside classical systems courses.

### 🔧 Tito CLI & Platform Reliability
- **UTF-8 everywhere**: Explicit UTF-8 encoding across module file I/O, CLI logging, release tools, test fixtures, SIMD compiler subprocesses, and book tools; `PYTHONUTF8=1` for child processes on Windows.
- **Student previews**: `tito module reset` and the default preview keep solution markers and the educational approach text; exercise stubs are cleaned.
- Module headers and export commands harmonized across `src/`; `benchmark.py` restored.
- CLI tagline and first-run logo animation match the book's subtitle.

## [0.1.17] — 2026-09-21

Patch release for the Jupyter launcher and the site version pill.

### 🔧 Tito CLI & Platform Reliability
- **Jupyter deadlock**: The `tito module` Jupyter Lab launcher redirects Jupyter Lab logs to a file instead of a pipe, preventing a hang once the pipe buffer fills (regression test added).
- **Version pill**: The sidebar version pill revalidates against the release manifest and adapts to narrow viewports.

## [0.1.16] — 2026-09-21

First tagged release after 0.1.14 (0.1.15 was documented but never tagged, so its changes ship here). Adds TinyGPT as the Milestone 05 headline and moves the reference GPT into a `tinytorch.models` package.

### 🧠 Core Framework & Autograd
- Includes the 0.1.15 `Linear` `requires_grad=True` change below.
- **Sequence losses**: `CrossEntropyLoss` and its backward accept `(..., num_classes)` logits with `(...)` targets; `Trainer` computes sequence-aware loss.
- **Module 13 (`transformers`)**: Keeps only the building blocks (`LayerNorm`, `MLP`, `TransformerBlock`, causal mask, sampling, generation); the reference `GPT` and `TinyGPT` live in `tinytorch.models.transformer`.
- **Optimizer** (M07): `Optimizer.__init__` no longer forces `requires_grad=True`, so frozen parameters stay frozen.
- **SIMD extension**: Validates matrix dimensions before the NumPy fallback.

### 🏆 Milestones
- **Milestone 05 (`transformer`)**: TinyGPT trained on TinyShakespeare (bundled offline sample) is the primary demonstration; attention sequence routing is Part 2. The briefly separate TinyGPT milestone was folded back in, keeping six milestones with monotonic unlocking.
- **Milestone 06 (`mlperf`)**: Split into three workload divisions (MLP, CNN, GPT), each with its own Pareto frontier, wired to the Module 19 benchmarking harness and measured metrics; scorecards simplified.

### 🔧 Tito CLI & Platform Reliability
- Fixed a cross-drive staging error when exporting modules on Windows; `install.sh` pins a default version.
- Release checks copy `tinytorch.models` into the progressive student-journey sandbox.
- ASCII charts replaced with SVGs; book listings regenerated for training and transformers.

## [0.1.15] — 2026-09-21

Patch release aligning layer initialization and autograd across all 20 modules. Learnable parameters in `Linear` (Module 03) are now initialized with `requires_grad=True` by default, bringing it into full parity with `Conv2d`, `BatchNorm2d`, `Embedding`, and `LayerNorm`. Removed downstream workaround loops in `Trainer`, milestone scripts, and test suites, enabling true out-of-the-box autograd recording and preserving parameter freezing during fine-tuning.

### 🧠 Core Framework & Autograd
- **Linear Parameter Invariant**: `Linear.weight` and `Linear.bias` now initialize with `requires_grad=True` at creation, matching PyTorch's `nn.Linear` and all subsequent TinyTorch layers (`Conv2d`, `Embedding`, `LayerNorm`).
- **Clean Trainer Loop**: Removed the workaround loop in `Trainer.__init__` that forcefully mutated model parameters, allowing user-configured parameter freezing (`param.requires_grad = False`) to persist during training.
- **Milestone Simplification**: Removed manual `requires_grad = True` patches in Milestone 04 (LeNet-5) and Milestone 05 (Vaswani Attention).
- **Test Suite Hygiene**: Cleaned up defensive `requires_grad` loops across all test suites (`06_autograd`, `09_convolutions`, `11_embeddings`, `12_attention`, `13_transformers`, `integration`, and `regression`) to verify authentic, unassisted gradient flow.

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
