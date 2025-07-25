
# MACHINE LEARNING SYSTEMS  
*Principles and Practices of Engineering Artificially Intelligent Systems*

[![Build Status](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/structure-check.yml?label=Build)](https://github.com/harvard-edge/cs249r_book/actions)
[![Website](https://img.shields.io/website?url=https://mlsysbook.ai&label=Website)](https://mlsysbook.ai)
[![Last Commit](https://img.shields.io/github/last-commit/harvard-edge/cs249r_book?label=Last%20Commit)](https://github.com/harvard-edge/cs249r_book/commits/dev)
[![License](https://img.shields.io/badge/license-CC--BY--NC--SA%204.0-blue)](https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE)
[![Open Collective](https://img.shields.io/badge/fund%20us-Open%20Collective-blue.svg)](https://opencollective.com/mlsysbook)

**Founded at Harvard University**

---

## 🎯 What Is This?

The **open-source textbook** that teaches you to build real-world AI systems — from edge devices to cloud deployment. Started as a Harvard course by Prof. Vijay Janapa Reddi, now used by universities and students worldwide.

> **Our mission:** Expand access to AI systems education worldwide — empowering learners, one chapter and one lab at a time.

For the full learning experience and instructor materials, including the textbook, hands-on labs, educational frameworks, kits, and community, please visit:  👉 [**https://mlsysbook.org**](https://mlsysbook.org)

---

## 📚 START HERE

### For Learners
- 📖 **[Read online](https://mlsysbook.ai)** — continuously updated version
- 📄 **[Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf)** — for offline access
- 🌐 **[Explore the full experience](https://mlsysbook.org)** — complete ecosystem

### For Educators  
- 🎓 **Course materials & labs** — hands-on learning resources
- 📋 **Instructor resources** — teaching guides and materials

### For Contributors
- 🤝 **[How to contribute](contribute.md)** — detailed guidelines
- ⚡ **[Quick setup](#quick-start)** — get started in minutes

---

## 🧠 About the Project

MLSysBook began as a Harvard course and has since grown into a **global educational movement** focused on teaching ML through a **systems-first lens**.

We go beyond training models — our goal is to help learners understand and build the full stack of real-world ML systems, from edge devices to cloud-scale deployment.

### Core Topics:
- ML system design & modularity  
- Data collection & labeling pipelines  
- Model architecture & optimization  
- Deployment, MLOps & monitoring  
- Edge AI & resource-constrained platforms

---

## 🚀 Quick Start

### For Readers
```bash
# View the book online
open https://mlsysbook.ai
```

### For Contributors
```bash
# Clone and setup
git clone https://github.com/harvard-edge/cs249r_book.git
cd cs249r_book
make setup-hooks  # Setup automated quality controls
make install      # Install dependencies

# Daily development
make clean build  # Clean and build
make preview      # Start development server
```

---

## 🤝 Contributing

We welcome contributions from around the world — from students, educators, researchers, and practitioners.

### Ways to Contribute
- 📝 **Content**: Suggest edits, improvements, or new examples
- 🛠️ **Tools**: Enhance development scripts and automation
- 🎨 **Design**: Improve figures, diagrams, and visual elements
- 🌍 **Localization**: Translate or adapt content for local needs
- 🔧 **Infrastructure**: Help with build systems and deployment

### Getting Started
1. **Read**: [contribute.md](contribute.md) for detailed guidelines
2. **Setup**: Follow the development workflow below
3. **Explore**: Check existing [GitHub Issues](https://github.com/harvard-edge/cs249r_book/issues)
4. **Connect**: Join [GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions)

### Quality Standards
All contributions go through automated quality checks:
- ✅ **Pre-commit validation**: Automatic cleanup and checks
- 📋 **Content review**: Formatting and style validation  
- 🧪 **Testing**: Automated build and link verification
- 👥 **Peer review**: Community and maintainer feedback
- 🗂️ **Organized workflow**: Professional development environment with clear tool categories

---

## 💡 Learn More

* 🌐 [mlsysbook.org](https://mlsysbook.org) — main site and learning platform
* 🔥 [TinyTorch](https://mlsysbook.org/tinytorch) — educational ML framework
* 💸 [Open Collective](https://opencollective.com/mlsysbook) — support this initiative
* 🧠 [GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions) — ask questions or share insights
* 📱 [Community](https://mlsysbook.org/community) — join our global learning community

---

<details>
<summary>🛠️ Development Workflow & Technical Details</summary>

## Development Workflow

This project features a **modern, automated development workflow** with quality controls and organized tooling.

### ⚡ Quick Commands

```bash
# Building
make build          # Build HTML version
make build-pdf      # Build PDF version  
make preview        # Start development server

# Quality Control
make clean          # Clean build artifacts
make test           # Run validation tests
make lint           # Check for issues
make check          # Project health check

# Get help
make help           # Show all commands
```

### 🔧 Automated Quality Controls

- **🧹 Pre-commit hooks**: Automatically clean build artifacts before commits
- **📋 Linting**: Check for formatting and content issues
- **✅ Validation**: Verify project structure and dependencies
- **🔍 Testing**: Automated tests for content and scripts
- **🗂️ Organized Structure**: Professional script organization with clear categories

### 🗂️ Organized Development Tools

Our development tools are organized into logical categories:

```
tools/scripts/
├── build/           # Build and development scripts
├── content/         # Content management tools
├── maintenance/     # System maintenance scripts
├── testing/         # Test and validation scripts
├── utilities/       # General utility scripts
└── docs/            # Comprehensive documentation
```

Each category includes focused tools with clear naming and documentation. See [`tools/scripts/README.md`](tools/scripts/README.md) for details.

## 📖 Documentation

- **📋 [DEVELOPMENT.md](DEVELOPMENT.md)** — Comprehensive development guide
- **🛠️ [MAINTENANCE_GUIDE.md](MAINTENANCE_GUIDE.md)** — Daily workflow and maintenance tasks
- **🔨 [BUILD.md](BUILD.md)** — Detailed build instructions  
- **🗂️ [tools/scripts/](tools/scripts/)** — Development tools documentation
- **🤝 [contribute.md](contribute.md)** — Contribution guidelines

## 🔧 Build the Book Locally

### Prerequisites
- [Quarto](https://quarto.org/docs/download/) (latest version)
- Python 3.8+ with pip
- Git

### Quick Build
```bash
# Clone the repository
git clone https://github.com/harvard-edge/cs249r_book.git
cd cs249r_book

# Setup development environment
make setup-hooks  # Configure git hooks
make install      # Install dependencies

# Build and preview (runs from book/ directory)
make clean build  # Clean and build HTML
make preview      # Start development server
```

### Advanced Development
```bash
# Full development setup
make clean-deep      # Deep clean
make install         # Install all dependencies
make build-all       # Build all formats (HTML, PDF, EPUB)

# Continuous development
make preview         # Auto-reload development server
make test            # Run validation tests
make lint            # Check content quality
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for the complete development guide.

## 📊 Project Structure

```
MLSysBook/
├── book/                    # Main book content (Quarto)
│   ├── contents/            # Chapter content
│   │   ├── core/            # Core chapters
│   │   ├── labs/            # Hands-on labs
│   │   ├── frontmatter/     # Preface, acknowledgments
│   │   └── parts/           # Book parts and sections
│   ├── _quarto.yml          # Book configuration
│   ├── index.qmd            # Main entry point
│   └── assets/              # Images, styles, media
├── tools/                   # Development automation
│   ├── scripts/             # Organized development scripts
│   │   ├── build/           # Build and development tools
│   │   ├── content/         # Content management tools
│   │   ├── maintenance/     # System maintenance scripts
│   │   ├── testing/         # Test and validation scripts
│   │   ├── utilities/       # General utility scripts
│   │   └── docs/            # Script documentation
│   ├── dependencies/        # Package requirements  
│   └── setup/               # Setup and configuration
├── config/                  # Build configuration
│   ├── _extensions/         # Quarto extensions
│   ├── lua/                 # Lua scripts
│   └── tex/                 # LaTeX templates
├── assets/                  # Global assets (covers, icons)
├── DEVELOPMENT.md           # Development guide
├── MAINTENANCE_GUIDE.md     # Daily workflow guide
├── Makefile                 # Development commands
└── README.md                # This file
```

## 🎯 Features

- **🚀 Modern Development Workflow**: Automated builds, quality checks, and deployment
- **🗂️ Organized Tooling**: Professional script organization with comprehensive documentation
- **🔧 Easy Contribution**: One-command setup with automated quality controls
- **📚 Comprehensive Docs**: Detailed guides for development, building, and contribution
- **🌐 Multi-format Output**: HTML, PDF, and EPUB with consistent styling
- **⚡ Fast Iteration**: Live preview server with automatic reloading
- **✅ Quality Assurance**: Automated testing, linting, and validation
- **📁 Clean Architecture**: Well-organized project structure with clear separation of concerns
- **🛠️ Professional Tooling**: Category-based script organization for easy maintenance

</details>

<details>
<summary>📋 Project Information</summary>

## 📖 Citation

```bibtex
@inproceedings{reddi2024mlsysbook,
  title        = {MLSysBook.AI: Principles and Practices of Machine Learning Systems Engineering},
  author       = {Reddi, Vijay Janapa},
  booktitle    = {2024 International Conference on Hardware/Software Codesign and System Synthesis (CODES+ ISSS)},
  pages        = {41--42},
  year         = {2024},
  organization = {IEEE},
  url          = {https://mlsysbook.org},
  note         = {Available at: https://mlsysbook.org}
}
```

## 🛡️ License

This work is licensed under a
**Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International License**
(CC BY-NC-SA 4.0)

You may share and adapt the material for **non-commercial purposes**, with appropriate credit and under the same license.

</details>

