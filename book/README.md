# Machine Learning Systems

*Principles and Practices of Engineering Artificially Intelligent Systems*

[![Build](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/book-validate-dev.yml?branch=dev&label=Build&logo=githubactions)](https://github.com/harvard-edge/cs249r_book/actions/workflows/book-validate-dev.yml)
[![Website](https://img.shields.io/badge/Read-mlsysbook.ai-blue)](https://mlsysbook.ai)
[![PDF](https://img.shields.io/badge/Download-PDF-red)](https://mlsysbook.ai/pdf)
[![EPUB](https://img.shields.io/badge/Download-EPUB-green)](https://mlsysbook.ai/epub)

**[Read Online](https://mlsysbook.ai)** | **[PDF](https://mlsysbook.ai/pdf)** | **[EPUB](https://mlsysbook.ai/epub)**

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

| Part | Focus | Chapters |
|------|-------|----------|
| **Foundations** | ML and systems basics | Introduction, ML Primer, DL Primer, AI Acceleration |
| **Workflow** | Production pipeline | Workflows, Data Engineering, Frameworks |
| **Training** | Learning at scale | Training, Distributed Training, Efficient AI |
| **Deployment** | Real-world systems | Inference, On-Device AI, Hardware Benchmarking, Ops |
| **Advanced** | Frontier topics | Privacy, Security, Responsible AI, Sustainable AI, Genertic AI, Frontiers |

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
open https://mlsysbook.ai

# Download formats
curl -O https://mlsysbook.ai/pdf
curl -O https://mlsysbook.ai/epub
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
│   │   ├── core/        # Core chapters
│   │   ├── labs/        # Hands-on labs
│   │   ├── frontmatter/ # Preface, about, changelog
│   │   └── backmatter/  # References, glossary
│   ├── assets/          # Images, downloads
│   └── _quarto.yml      # Quarto configuration
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
| **Readers** | [Online Book](https://mlsysbook.ai) ・ [PDF](https://mlsysbook.ai/pdf) ・ [EPUB](https://mlsysbook.ai/epub) |
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
| **[Website](https://mlsysbook.ai)** | Read the book online |

---

## Contributors

Thanks to these wonderful people who helped improve the book ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=80" width="80px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🐛 💻 🎨 📖 💡 👀 🧪 🔧</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Mjrovai"><img src="https://avatars.githubusercontent.com/Mjrovai?v=4?s=80" width="80px;" alt="Marcelo Rovai"/><br /><sub><b>Marcelo Rovai</b></sub></a><br />💻 🎨 🧪</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/GabrielAmazonas"><img src="https://avatars.githubusercontent.com/GabrielAmazonas?v=4?s=80" width="80px;" alt="Gabriel Amazonas"/><br /><sub><b>Gabriel Amazonas</b></sub></a><br />🐛 📖 💡</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hzeljko"><img src="https://avatars.githubusercontent.com/hzeljko?v=4?s=80" width="80px;" alt="Zeljko Hrcek"/><br /><sub><b>Zeljko Hrcek</b></sub></a><br />💻</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

**Recognize a contributor:** Comment on any issue or PR:
```
@all-contributors please add @username for doc, review, translation, or design
```

---

## License

Book content is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (CC BY-NC-SA 4.0).

See [LICENSE.md](../LICENSE.md) for details.
