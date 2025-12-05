# Machine Learning Systems - Book

*Principles and Practices of Engineering Artificially Intelligent Systems*

This directory contains the MLSysBook textbook content and build system.

For the full project overview including TinyTorch, see the [root README](../README.md).

## Quick Start

```bash
# First time setup
./binder setup
./binder doctor

# Daily workflow
./binder clean
./binder build
./binder preview intro

# Build all formats
./binder pdf
./binder epub
```

## Directory Structure

```
book/
├── quarto/          # Book source (Quarto markdown)
│   ├── contents/    # Chapter content
│   ├── assets/      # Images, downloads
│   └── config/      # Quarto configurations
├── cli/             # Binder CLI tool
├── docker/          # Development containers
├── docs/            # Documentation
├── tools/           # Build scripts
└── binder           # CLI entry point
```

## Development

For detailed contribution guidelines, build instructions, and development workflows, see:

- [docs/contribute.md](./docs/contribute.md) - How to contribute
- [docs/BUILD.md](./docs/BUILD.md) - Build system details
- [docs/DEVELOPMENT.md](./docs/DEVELOPMENT.md) - Development guide
- [docs/BINDER.md](./docs/BINDER.md) - CLI documentation

## License

CC BY-NC-SA 4.0 - See LICENSE.md

## Website

https://mlsysbook.ai
