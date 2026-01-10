# Machine Learning Systems

*Principles and Practices of Engineering Artificially Intelligent Systems*

[![Build](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/book-validate-dev.yml?branch=dev&label=Build&logo=githubactions)](https://github.com/harvard-edge/cs249r_book/actions/workflows/book-validate-dev.yml)
[![Website](https://img.shields.io/badge/Read-mlsysbook.ai-blue)](https://mlsysbook.ai/book/)
[![PDF](https://img.shields.io/badge/Download-PDF-red)](https://mlsysbook.ai/book/assets/downloads/Machine-Learning-Systems.pdf)
[![EPUB](https://img.shields.io/badge/Download-EPUB-green)](https://mlsysbook.ai/book/assets/downloads/Machine-Learning-Systems.epub)

**[Read Online](https://mlsysbook.ai/book/)** | **[Volume I](https://mlsysbook.ai/vol1/)** | **[Volume II](https://mlsysbook.ai/vol2/)** | **[PDF](https://mlsysbook.ai/book/assets/downloads/Machine-Learning-Systems.pdf)**

---

## What This Is

The ML Systems textbook teaches you how to engineer AI systems that work in the real world. It bridges machine learning theory with systems engineering practice, covering everything from neural network fundamentals to production deployment.

This directory contains the textbook source and build system for contributors.

---

## What You Will Learn

| ML Concepts | Systems Engineering |
|-------------|---------------------|
| Neural networks and deep learning | Memory hierarchies and caching |
| Model architectures (CNNs, Transformers) | Hardware accelerators (GPUs, TPUs, NPUs) |
| Training and optimization | Distributed systems and parallelism |
| Inference and deployment | Power and thermal management |
| Compression and quantization | Latency, throughput, and efficiency |

### The ML ↔ Systems Bridge

| You know... | You will learn... |
|-------------|-------------------|
| How to train a model | How training scales across GPU clusters |
| That quantization shrinks models | How INT8 math maps to silicon |
| What a transformer is | Why KV-cache dominates memory |
| Models run on GPUs | How schedulers balance latency vs throughput |
| Edge devices have limits | How to co-design models and hardware |

### Book Structure

This textbook is organized into **two volumes** following the Hennessy & Patterson pedagogical model:

| Volume | Theme | Focus |
|--------|-------|-------|
| **Volume I** | Build, Optimize, Deploy | Single-machine ML systems, foundational principles |
| **Volume II** | Scale, Distribute, Govern | Distributed systems at production scale |

#### Volume I: Build, Optimize, Deploy

| Part | Focus | Chapters |
|------|-------|----------|
| **Foundations** | Core concepts | Introduction, ML Systems, DL Primer, Architectures |
| **Development** | Building blocks | Workflow, Data Engineering, Frameworks, Training |
| **Optimization** | Making it fast | Efficient AI, Optimizations, HW Acceleration, Benchmarking |
| **Deployment** | Making it work | Serving, MLOps, Responsible Engineering |

#### Volume II: Scale, Distribute, Govern

| Part | Focus | Chapters |
|------|-------|----------|
| **Foundations of Scale** | Infrastructure | Infrastructure, Storage, Communication |
| **Distributed Systems** | Coordination | Distributed Training, Fault Tolerance, Inference, Edge Intelligence |
| **Production Challenges** | Operations | On-device Learning, Privacy & Security, Robust AI, Ops at Scale |
| **Responsible Deployment** | Trust | Responsible AI, Sustainable AI, AI for Good, Frontiers |

---

## What Makes This Book Different

**Systems first**: Start with hardware constraints and work up to algorithms, not the other way around.

**Production focus**: Every concept connects to real deployment scenarios, not just research benchmarks.

**Open and evolving**: Community-driven updates keep content current with a fast-moving field.

**Hands-on companion**: Pair with [TinyTorch](../tinytorch/) to build what you learn from scratch.

---

## Quick Start

### For Readers

```bash
# Read online
open https://mlsysbook.ai/book/

# Download formats
curl -O https://mlsysbook.ai/book/assets/downloads/Machine-Learning-Systems.pdf
curl -O https://mlsysbook.ai/book/assets/downloads/Machine-Learning-Systems.epub
```

### For Contributors

```bash
cd book

# First time setup
./binder setup
./binder doctor

# Daily workflow
./binder clean              # Clean build artifacts
./binder build              # Build HTML book
./binder preview intro      # Preview chapter with live reload

# Build all formats
./binder pdf                # Build PDF
./binder epub               # Build EPUB

# Utilities
./binder help               # Show all commands
./binder list               # List chapters
```

---

## Directory Structure

```
book/
├── quarto/              # Book source (Quarto markdown)
│   ├── contents/        # Chapter content
│   │   ├── vol1/        # Volume I: Build, Optimize, Deploy
│   │   ├── vol2/        # Volume II: Scale, Distribute, Govern
│   │   ├── frontmatter/ # Preface, about, changelog
│   │   └── backmatter/  # References, glossary
│   ├── assets/          # Images, downloads
│   └── config/          # Quarto configuration files
├── cli/                 # Binder CLI tool
├── docker/              # Development containers
├── docs/                # Documentation
├── tools/               # Build scripts
└── binder               # CLI entry point
```

---

## Documentation

| Audience | Resources |
|----------|-----------|
| **Readers** | [Online Book](https://mlsysbook.ai/book/) ・ [Volume I](https://mlsysbook.ai/vol1/) ・ [Volume II](https://mlsysbook.ai/vol2/) ・ [PDF](https://mlsysbook.ai/book/assets/downloads/Machine-Learning-Systems.pdf) |
| **Contributors** | [CONTRIBUTING.md](docs/CONTRIBUTING.md) ・ [BUILD.md](docs/BUILD.md) |
| **Developers** | [DEVELOPMENT.md](docs/DEVELOPMENT.md) ・ [BINDER.md](docs/BINDER.md) |

---

## Contributing

We welcome contributions! See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

1. **Fork and clone** the repository
2. **Set up** your environment: `./binder setup`
3. **Find an issue** or propose a change
4. **Make your changes** in the `quarto/contents/` directory
5. **Preview** your changes: `./binder preview <chapter>`
6. **Submit a PR** with a clear description

---

## Related

| Component | Description |
|-----------|-------------|
| **[Main README](../README.md)** | Project overview and ecosystem |
| **[TinyTorch](../tinytorch/)** | Build ML frameworks from scratch |
| **[Hardware Kits](../kits/)** | Deploy to Arduino, Raspberry Pi, edge devices |
| **[Website](https://mlsysbook.ai/book/)** | Read the book online |

---

## License

Book content is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (CC BY-NC-SA 4.0).

See [LICENSE.md](../LICENSE.md) for details.
