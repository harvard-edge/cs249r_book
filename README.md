# MACHINE LEARNING SYSTEMS  
*Principles and Practices of Engineering Artificially Intelligent Systems*

[![Build Status](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/controller.yml?branch=dev&label=Build)](https://github.com/harvard-edge/cs249r_book/actions/workflows/controller.yml?query=branch%3Adev)
[![📖 Website](https://img.shields.io/website?url=https://mlsysbook.ai&label=Website)](https://mlsysbook.ai)
[![🌐 Ecosystem](https://img.shields.io/website?url=https://mlsysbook.org&label=Ecosystem)](https://mlsysbook.org)
[![Last Commit](https://img.shields.io/github/last-commit/harvard-edge/cs249r_book?label=Last%20Commit)](https://github.com/harvard-edge/cs249r_book/commits/dev)
[![License](https://img.shields.io/badge/license-CC--BY--NC--SA%204.0-blue)](https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE)
[![Open Collective](https://img.shields.io/badge/fund%20us-Open%20Collective-blue.svg)](https://opencollective.com/mlsysbook)

---

## 🎯 What Is This?

The **open-source textbook** that teaches you to build real-world AI systems — from edge devices to cloud deployment. Started as a Harvard University course (CS249r) by [Prof. Vijay Janapa Reddi](https://github.com/profvjreddi/homepage), now used by universities and students worldwide.

> **Our mission:** Expand access to AI systems education worldwide — empowering learners, one chapter and one lab at a time.

## 💭 Why This Exists

*"This grew out of a concern that while students could train AI models, few understood how to build the systems that actually make them work. It's like everyone can write an app, but few know how to build the smartphone that runs it. As AI becomes more capable and autonomous, the critical bottleneck won't be the algorithms - it will be the engineers who can build efficient, scalable, and sustainable systems that safely harness that intelligence. Richard Sutton's "The Bitter Lesson" taught us that general methods leveraging computation ultimately triumph over human-crafted approaches. The same principle applies here: the future belongs to those who can engineer the systems that unlock AI's computational potential. We're at an inflection point where we need an entirely new discipline - AI Engineering - focused not just on training models, but on the full systems stack that makes AI work in the real world. This book is my attempt to establish that foundation. We can't make this transformation happen overnight, but it has to start somewhere."*

— *Vijay*

### ⭐ Show Community Support
**Your star proves to funders this educational resource matters.**

📊 [![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=social&label=Community%20support)](https://github.com/harvard-edge/cs249r_book/stargazers) demonstrate global value  
🎯 **Goal:** 10,000 stars = $100,000 in additional education funding

**[⭐ Star this project](https://github.com/harvard-edge/cs249r_book)** - takes 2 seconds!
---

## 📚 GET STARTED

### 🎓 For Learners
- 📖 **[Read online](https://mlsysbook.ai)** — continuously updated version
- 📄 **[Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf)** — for offline access
- 🌐 **[Explore the ecosystem](https://mlsysbook.org)** — complete learning experience with labs & frameworks

### 👩‍🏫 For Educators  
- 🎓 **[Course materials & labs](https://mlsysbook.org)** — hands-on learning resources
- 📋 **[Instructor resources](https://mlsysbook.org)** — teaching guides and materials

### 🛠️ For Contributors
- 🤝 **[How to contribute](contribute.md)** — detailed guidelines
- ⚡ **[Quick setup](#quick-start)** — get started in minutes

---

## 🧠 What You'll Learn

We go beyond training models — this book teaches you to understand and build the **full stack** of real-world ML systems.

**Core Topics:**
- **ML system design & architecture** — building scalable, maintainable systems
- **Data pipelines & engineering** — collection, labeling, and processing at scale  
- **Model optimization & deployment** — from prototypes to production
- **MLOps & monitoring** — keeping systems running reliably
- **Edge AI & resource constraints** — deploying on mobile, embedded, and IoT devices

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

We welcome contributions from students, educators, researchers, and practitioners worldwide.

### Ways to Contribute
- 📝 **Content**: Suggest edits, improvements, or new examples
- 🛠️ **Tools**: Enhance development scripts and automation
- 🎨 **Design**: Improve figures, diagrams, and visual elements
- 🌍 **Localization**: Translate or adapt content for local needs
- 🔧 **Infrastructure**: Help with build systems and deployment

### Getting Started
1. **Read**: [contribute.md](docs/contribute.md) for detailed guidelines
2. **Setup**: Follow the [development workflow](#quick-start) above
3. **Explore**: Check existing [GitHub Issues](https://github.com/harvard-edge/cs249r_book/issues)
4. **Connect**: Join [GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions)

### Quality Standards
All contributions go through automated quality checks:
- ✅ **Pre-commit validation**: Automatic cleanup and checks
- 📋 **Content review**: Formatting and style validation  
- 🧪 **Testing**: Automated build and link verification
- 👥 **Peer review**: Community and maintainer feedback

---

## ⭐ Support This Work

**Show this matters:** If you find this valuable, please **star this repository** ⭐ — it signals to institutions and funding bodies that open AI education matters.

**Fund the mission:** Help us expand AI systems education globally. You can sponsor TinyML kits for students in developing countries, fund learning materials, support workshops, or sustain our open-source infrastructure.

[![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg)](https://opencollective.com/mlsysbook)

From $15/month to sponsor a learner, to $250 for a hands-on workshop — every contribution democratizes AI systems education worldwide.

---

## 🌐 Learn More

* 📚 [mlsysbook.org](https://mlsysbook.org) — main site and learning platform
* 🔥 [TinyTorch](https://mlsysbook.org/tinytorch) — educational ML framework
* 💸 [Open Collective](https://opencollective.com/mlsysbook) — support this initiative
* 🧠 [GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions) — ask questions or share insights
* 📱 [Community](https://mlsysbook.org/community) — join our global learning community

---

## 🛠️ Development Workflow & Technical Details

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

- **📋 [DEVELOPMENT.md](docs/DEVELOPMENT.md)** — Comprehensive development guide
- **🛠️ [MAINTENANCE_GUIDE.md](docs/MAINTENANCE_GUIDE.md)** — Daily workflow and maintenance tasks
- **🔨 [BUILD.md](docs/BUILD.md)** — Detailed build instructions  
- **🗂️ [tools/scripts/](tools/scripts/)** — Development tools documentation
- **🤝 [contribute.md](docs/contribute.md)** — Contribution guidelines

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

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for the complete development guide.

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

---

## 📋 Project Information

### 📖 Citation

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

### 🛡️ License

This work is licensed under a
**Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International License**
(CC BY-NC-SA 4.0)

You may share and adapt the material for **non-commercial purposes**, with appropriate credit and under the same license.

