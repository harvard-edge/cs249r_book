# Machine Learning Systems
*Principles and Practices of Engineering Artificially Intelligent Systems*

<div align="center">
  
<p align="center">

  <!-- Row 1: Project Health -->
  [![Build](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/validate-dev.yml?branch=dev&label=Build&logo=githubactions&cacheSeconds=300)](https://github.com/harvard-edge/cs249r_book/actions/workflows/validate-dev.yml)
  ![Last Commit](https://img.shields.io/github/last-commit/harvard-edge/cs249r_book/dev?label=Last%20Commit&logo=git&cacheSeconds=300)

</p>

<p align="center">

  <!-- Row 2: Access & Ecosystem -->
  [![Website](https://img.shields.io/website?url=https%3A%2F%2Fmlsysbook.ai&label=Website&logo=readthedocs)](https://mlsysbook.ai)
  [![Ecosystem](https://img.shields.io/website?url=https%3A%2F%2Fmlsysbook.org&label=Ecosystem&logo=internet-explorer)](https://mlsysbook.org)
  [![Paper](https://img.shields.io/badge/Paper-MLSysBook.AI%20Overview-blue?logo=academia)](LINK_TO_PAPER)

</p>

<p align="center">

  <!-- Row 3: Support -->
  [![Funding](https://img.shields.io/badge/Fund%20Us-Open%20Collective-blue.svg?logo=open-collective)](https://opencollective.com/mlsysbook)
  [![License](https://img.shields.io/badge/License-CC--BY--NC--SA%204.0-blue.svg)](https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE)
  [![Powered by Netlify](https://img.shields.io/badge/Powered%20by-Netlify-00C7B7?logo=netlify&logoColor=white)](https://www.netlify.com)

</p>

<p align="center">

  <!-- Reader Navigation -->
  **[📖 Read Online](https://mlsysbook.ai)** • 
  **[💾 Download PDF](https://mlsysbook.ai/pdf)** • 
  **[💾 Download ePub](https://mlsysbook.ai/epub)** • 
  **[🌐 Explore Ecosystem](https://mlsysbook.org)**

</p>

📚 **Hardcopy edition coming 2026 via MIT Press!**

</div>

---

## About This Book

The **open-source textbook** that teaches you to build real-world AI systems — from edge devices to cloud deployment. Originally developed as Harvard University's CS249r course by [Prof. Vijay Janapa Reddi](https://github.com/profvjreddi/homepage), now used by universities and students worldwide.

> **Our mission:** Expand access to AI systems education worldwide — empowering learners, one chapter and one lab at a time.

### Why This Book Exists

*"This grew out of a concern that while students could train AI models, few understood how to build the systems that actually make them work. As AI becomes more capable and autonomous, the critical bottleneck won't be the algorithms - it will be the engineers who can build efficient, scalable, and sustainable systems that safely harness that intelligence."*

**— Vijay Janapa Reddi**

---

## 📚 What You'll Learn

Go beyond training models — master the **full stack** of real-world ML systems.

| Topic | What You'll Build |
|-------|------------------|
| **System Design** | Scalable, maintainable ML architectures |
| **Data Engineering** | Robust pipelines for collection, labeling, and processing |
| **Model Deployment** | Production-ready systems from prototypes |
| **MLOps & Monitoring** | Reliable, continuously operating systems |
| **Edge AI** | Resource-efficient deployment on mobile, embedded, and IoT |

---

## ⭐ Support This Work

<div align="center">

### Show Your Support
**Star this repository** to help us demonstrate the value of open AI education to funders and institutions.

[![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)

**Goal:** 10,000 stars = $100,000 in additional education funding

[**⭐ Star Now**](https://github.com/harvard-edge/cs249r_book) — *takes 2 seconds!*

### Fund the Mission (New!)
We've graduated this project from Harvard to enable global access and expand AI systems education worldwide. Please help us support educators globally, especially in the Global South, by providing TinyML kits for students, funding workshops, and sustaining our open-source infrastructure.

[![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

*From $15/month to sponsor a learner to $250 for workshops — every contribution democratizes AI education.*

</div>

---

## 🌐 Community & Resources

| Resource | Description |
|----------|-------------|
| [📚 **Main Site**](https://mlsysbook.org) | Complete learning platform |
| [🔥 **TinyTorch**](https://mlsysbook.org/tinytorch) | Educational ML framework |
| [💬 **Discussions**](https://github.com/harvard-edge/cs249r_book/discussions) | Ask questions, share insights |
| [👥 **Community**](https://mlsysbook.org/community) | Join our global learning community |

---

## 🎯 For Different Audiences

### 🎓 Students
- [📖 Read online](https://mlsysbook.ai)
- [📄 Download PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf)
- [🧪 Try hands-on labs](https://mlsysbook.org)

### 👩‍🏫 Educators
- [📋 Course materials](https://mlsysbook.org)
- [🎯 Instructor resources](https://mlsysbook.org)
- [💡 Teaching guides](https://mlsysbook.org)

### 🛠️ Contributors
- [🤝 Contribution guide](docs/contribute.md)
- [⚡ Development setup](#development)
- [💬 Join discussions](https://github.com/harvard-edge/cs249r_book/discussions)

---

## 🚀 Quick Start

### For Readers
```bash
# Read online (continuously updated)
open https://mlsysbook.ai

# Or download PDF for offline access
curl -O https://mlsysbook.ai/Machine-Learning-Systems.pdf
```

### For Contributors
```bash
git clone https://github.com/harvard-edge/cs249r_book.git
cd cs249r_book

# Quick setup (recommended)
./binder setup      # Setup environment and dependencies
./binder doctor     # Check system health

# Fast development workflow
./binder preview intro    # Fast chapter development
./binder build intro      # Build specific chapter
./binder build            # Build complete book (HTML)
./binder help            # See all commands
```

---

## 🤝 Contributing

We welcome contributions from the global community! Here's how you can help:

### Ways to Contribute
- **📝 Content** — Suggest edits, improvements, or new examples
- **🛠️ Tools** — Enhance development scripts and automation  
- **🎨 Design** — Improve figures, diagrams, and visual elements
- **🌍 Localization** — Translate content for global accessibility
- **🔧 Infrastructure** — Help with build systems and deployment

### Quality Standards
All contributions benefit from automated quality assurance:
- ✅ **Pre-commit validation** — Automatic cleanup and checks
- 📋 **Content review** — Formatting and style validation
- 🧪 **Testing** — Build and link verification
- 👥 **Peer review** — Community feedback

[**Start Contributing →**](docs/contribute.md)

---

## 🛠️ Development

### Book Binder CLI (Recommended)

The **Book Binder** is our lightning-fast development CLI for streamlined building and iteration:

```bash
# Chapter development (fast iteration)
./binder preview intro                # Build and preview single chapter
./binder preview intro,ml_systems     # Build and preview multiple chapters

# Complete book building
./binder build                        # Build complete website (HTML)
./binder pdf                          # Build complete PDF
./binder epub                         # Build complete EPUB

# Management
./binder clean                        # Clean artifacts
./binder status                       # Show current status
./binder doctor                       # Run health check
./binder help                         # Show all commands
```

### Development Commands
```bash
# Book Binder CLI (Recommended)
./binder setup            # First-time setup
./binder build            # Build complete HTML book
./binder pdf              # Build complete PDF book  
./binder epub             # Build complete EPUB book
./binder preview intro    # Preview chapter development

# Traditional setup (if needed)
python3 -m venv .venv
source .venv/bin/activate
pip install -r tools/dependencies/requirements.txt
pre-commit install
```

### Project Structure
```
MLSysBook/
├── binder                   # ⚡ Fast development CLI (recommended)
├── quarto/                  # Main book content (Quarto)
│   ├── contents/            # Chapter content
│   │   ├── core/            # Core chapters
│   │   ├── labs/            # Hands-on labs
│   │   ├── frontmatter/     # Preface, acknowledgments
│   │   ├── backmatter/      # References and resources
│   │   └── parts/           # Book parts and sections
│   ├── _extensions/         # Quarto extensions
│   ├── config/              # Build configurations
│   │   ├── _quarto-html.yml # Website build configuration
│   │   └── _quarto-pdf.yml  # PDF build configuration
│   ├── data/                # Cross-reference and metadata files
│   ├── assets/              # Images, styles, media
│   ├── filters/             # Lua filters
│   ├── scripts/             # Build scripts
│   └── _quarto.yml          # Active config (symlink)
├── tools/                   # Development automation
│   ├── scripts/             # Organized development scripts
│   │   ├── content/         # Content management tools
│   │   ├── cross_refs/      # Cross-reference management
│   │   ├── genai/           # AI-assisted content tools
│   │   ├── maintenance/     # System maintenance scripts
│   │   ├── testing/         # Test and validation scripts
│   │   └── utilities/       # General utility scripts
│   ├── dependencies/        # Package requirements  
│   └── setup/               # Setup and configuration
├── config/                  # Project configuration
│   ├── dev/                 # Development configurations
│   ├── linting/             # Code quality configurations
│   └── quarto/              # Quarto publishing settings
├── docs/                    # Documentation
│   ├── BINDER.md            # Binder CLI guide
│   ├── BUILD.md             # Build instructions
│   ├── DEVELOPMENT.md       # Development guide
│   └── contribute.md        # Contribution guidelines
├── CHANGELOG.md             # Project changelog
├── CITATION.bib             # Citation information
├── pyproject.toml           # Python project configuration
└── README.md                # This file
```

### Documentation
- [⚡ Binder CLI Guide](docs/BINDER.md) — Fast development with the Book Binder
- [📋 Development Guide](docs/DEVELOPMENT.md) — Comprehensive setup and workflow
- [🛠️ Maintenance Guide](docs/MAINTENANCE_GUIDE.md) — Daily tasks and troubleshooting  
- [🔨 Build Instructions](docs/BUILD.md) — Detailed build process
- [🤝 Contribution Guidelines](docs/contribute.md) — How to contribute effectively

### Publishing
```bash
# Interactive publishing (recommended)
./binder publish

# Command-line publishing
./binder publish "Description" COMMIT_HASH

# Manual workflow (if needed)
./binder build html && ./binder build pdf
# Then use GitHub Actions to deploy
```

**Publishing Options:**
- **`./binder publish`** — Unified command with interactive and command-line modes
- **GitHub Actions** — Automated deployment via workflows

### Getting Started
```bash
# First time setup
./binder setup

# Check system health
./binder doctor

# Quick preview
./binder preview intro
```

---

## 📋 Citation & License

### Citation
```bibtex
@inproceedings{reddi2024mlsysbook,
  title        = {MLSysBook.AI: Principles and Practices of Machine Learning Systems Engineering},
  author       = {Reddi, Vijay Janapa},
  booktitle    = {2024 International Conference on Hardware/Software Codesign and System Synthesis (CODES+ ISSS)},
  pages        = {41--42},
  year         = {2024},
  organization = {IEEE},
  url          = {https://mlsysbook.org}
}
```

### License
This work is licensed under **Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International** (CC BY-NC-SA 4.0). You may share and adapt the material for non-commercial purposes with appropriate credit.

---

## 🙏 Contributors

Thanks goes to these wonderful people who have contributed to making this resource better for everyone:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

---

<div align="center">

**Made with ❤️ for AI learners worldwide**

Our goal is to educate 1 million AI systems engineers for the future at the edge of AI.
</div>
# Trigger build
