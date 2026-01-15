# Changelog

All notable changes to TinyTorch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Dynamic `tito --version` command showing current TinyTorch version
- CHANGELOG.md for tracking releases
- Updated publish workflow with release_type (patch/minor/major)

### Changed
- Version now managed in `tinytorch/__init__.py` and `pyproject.toml`

## [0.1.1] - 2025-01-13

### Fixed
- Module 03 (Layers): Removed premature `requires_grad` from `Linear` layer initialization
  - Aligns with progressive disclosure model where `requires_grad` is introduced in Module 06
  - Fixes issue where students running modules in sequence encountered undefined parameters

### Added
- `tinydigits` dataset: 8x8 handwritten digits for educational CNN training
- `tinytalks` dataset: Q&A pairs for transformer training examples

## [0.1.0] - 2024-12-12

### Added
- Initial public release of TinyTorch
- 20 progressive modules covering ML fundamentals to advanced topics
- `tito` CLI for guided learning experience
- Milestone projects demonstrating historical ML breakthroughs
- Comprehensive test suite
- Jupyter Book documentation site

### Modules
- 01: Tensor (NumPy wrapper with ML semantics)
- 02: Activations (Sigmoid, ReLU, Tanh, GELU, Softmax)
- 03: Layers (Linear, Dropout)
- 04: Losses (MSE, CrossEntropy)
- 05: DataLoader (batching, shuffling)
- 06: Autograd (automatic differentiation)
- 07: Optimizers (SGD, Adam)
- 08: Training (training loops)
- 09: Convolutions (Conv2D, pooling)
- 10: Tokenization (BPE, character level)
- 11: Embeddings (word embeddings)
- 12: Attention (self attention, multi head)
- 13: Transformers (encoder, decoder)
- 14: Profiling (timing, memory)
- 15: Quantization (INT8, dynamic)
- 16: Compression (pruning, distillation)
- 17: Acceleration (SIMD, parallelism)
- 18: Memoization (caching, checkpointing)
- 19: Benchmarking (MLPerf style)
- 20: Capstone (integration project)

[Unreleased]: https://github.com/harvard-edge/cs249r_book/compare/tinytorch-v0.1.1...HEAD
[0.1.1]: https://github.com/harvard-edge/cs249r_book/compare/tinytorch-v0.1.0...tinytorch-v0.1.1
[0.1.0]: https://github.com/harvard-edge/cs249r_book/releases/tag/tinytorch-v0.1.0
