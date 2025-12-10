# Machine Learning Systems - Book

*Build instructions for contributors*

[![Book](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/book-validate-dev.yml?branch=dev&label=Build&logo=githubactions)](https://github.com/harvard-edge/cs249r_book/actions/workflows/book-validate-dev.yml)
[![Website](https://img.shields.io/badge/Read-mlsysbook.ai-blue)](https://mlsysbook.ai)

This directory contains the MLSysBook textbook source and build system.

**[📖 Read Online](https://mlsysbook.ai)** • **[📄 PDF](https://mlsysbook.ai/pdf)** • **[📓 EPUB](https://mlsysbook.ai/epub)**

---

## Quick Start

```bash
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

## Contributing

1. **Fork and clone** the repository
2. **Set up** your environment: `./binder setup`
3. **Find an issue** or propose a change
4. **Make your changes** in the `quarto/contents/` directory
5. **Preview** your changes: `./binder preview <chapter>`
6. **Submit a PR** with a clear description

### Documentation

- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) - Contribution guide
- [docs/BUILD.md](docs/BUILD.md) - Build system details
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) - Development guide
- [docs/BINDER.md](docs/BINDER.md) - CLI documentation

---

## Related

- **[Root README](../README.md)** - Project overview and what you will learn
- **[TinyTorch](../tinytorch/)** - Hands-on ML framework companion
- **[Website](https://mlsysbook.ai)** - Read the book online

---

## License

Book content is licensed under **Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International** (CC BY-NC-SA 4.0).

See [LICENSE.md](../LICENSE.md) for details.
