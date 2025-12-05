# MLSysBook & TinyTorch

This repository contains two interconnected projects for AI systems education:

## 📚 [MLSysBook](./book/) - Machine Learning Systems Textbook

An open-source textbook for learning how to engineer AI systems.

- **Website**: https://mlsysbook.ai
- **Directory**: [`book/`](./book/)
- **Topics**: ML Systems, Edge AI, TinyML, Production ML
- **Format**: HTML, PDF, EPUB

### Quick Start (Book)

```bash
cd book
./binder setup
./binder build
./binder preview
```

## 🔥 [TinyTorch](./tinytorch/) - Educational ML Framework *(Coming Soon)*

A minimal PyTorch-like framework for learning deep learning internals through hands-on implementation.

- **Website**: https://tinytorch.ai (coming soon)
- **Directory**: [`tinytorch/`](./tinytorch/) (to be added)
- **Approach**: Progressive 20-module course building a complete ML framework from scratch

### Quick Start (TinyTorch)

```bash
# Coming after repository restructuring is complete
cd tinytorch
pip install -e .
python -m pytest tests/
```

## 🏗️ Repository Structure

```
MLSysBook/
├── book/                  # MLSysBook textbook
│   ├── quarto/           # Book source files (Quarto)
│   ├── cli/              # Build tools (binder CLI)
│   ├── docker/           # Development containers
│   └── docs/             # Documentation
├── tinytorch/            # TinyTorch framework (to be added)
│   ├── tinytorch/        # Core library
│   ├── modules/          # 20 learning modules
│   ├── tito/             # CLI tool
│   └── tests/            # Test suite
└── .github/              # Shared CI/CD workflows
```

## 🤝 Contributing

- **Book contributions**: See [book/docs/contribute.md](./book/docs/contribute.md)
- **TinyTorch contributions**: Coming soon

## 📄 License

- **MLSysBook**: CC BY-NC-SA 4.0 (see [book/LICENSE.md](./book/LICENSE.md))
- **TinyTorch**: MIT License (coming soon)

## 🌐 Community

- **Website**: https://mlsysbook.ai
- **Discussions**: https://github.com/harvard-edge/cs249r_book/discussions
- **Issues**: https://github.com/harvard-edge/cs249r_book/issues

---

**Note**: This repository is currently being restructured to support both MLSysBook and TinyTorch in a unified codebase. The `book/` directory contains all existing MLSysBook content, and the `tinytorch/` directory will be added soon.

Made with ❤️ for AI learners worldwide.
