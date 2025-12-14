# 🛠 How to Build the Book Locally

Welcome! 👋 If you’re here, you’re probably trying to **build the Machine Learning Systems book locally** on your own machine.

This guide will walk you through **how to get set up manually**, especially if you're not using GitHub Actions or Docker. We'll cover what tools you need, why you need them, and how to test everything is working.

## 🚀 Quick Start (Recommended)

For most users, the easiest way is using our **Book Binder CLI**:

```bash
# First time setup
./binder setup

# System health check
./binder doctor

# Quick chapter preview (HTML with live reload)
./binder preview intro

# Build specific chapter(s)
./binder build intro                    # Single chapter (HTML)
./binder build intro,ml_systems         # Multiple chapters (HTML)

# Build complete book
./binder build                          # Complete book (HTML)
./binder pdf                            # Complete book (PDF)
./binder epub                           # Complete book (EPUB)

# Get help
./binder help
```

The `binder` tool automatically handles all dependencies, configuration, and build processes for you!

---

## 🔧 Manual Setup (Advanced)

## 📚 What Are We Trying to Build?

This project is written using [**Quarto**](https://quarto.org), which lets us render:

- A website (HTML version of the book)
- A typeset PDF (for printable reading)

By default, Quarto can build the HTML version pretty easily. But **building the PDF version** is a bit trickier — it requires LaTeX, Inkscape, and a few other tools to properly render graphics and fonts.

---

## ✅ What You'll Need (And Why)

| Tool | Why It's Needed | Version |
|------|------------------|---------|
| **Quarto** | The core tool that converts the `.qmd` files into HTML/PDF | 1.7.31+ |
| **Python** | Required for Book Binder CLI and build scripts | 3.9+ |
| **Python packages** | Dependencies (see `tools/dependencies/requirements.txt`) | See below |
| **R** | Some chapters include R code chunks and R-based plots | 4.0+ |
| **R packages** | Supporting packages (defined in `tools/dependencies/install_packages.R`) | Latest |
| **TinyTeX + TeX Live** | Needed for LaTeX → PDF rendering | Latest |
| **Inkscape** | Converts `.svg` diagrams into `.pdf` (especially TikZ) | 1.0+ |
| **Ghostscript** | Compresses large PDF files | Latest |
| **System libraries** | Fonts and rendering support (Linux systems) | Various |

Don't worry — this guide will walk you through installing all of them, step by step.

### Python Dependencies

The project uses a modern Python packaging setup with `pyproject.toml`. Core dependencies include:

**Core Build Dependencies:**
- `jupyterlab-quarto>=0.3.0` - Quarto integration
- `jupyter>=1.0.0` - Jupyter notebook support
- `pybtex>=0.24.0` - Bibliography processing
- `pypandoc>=1.11` - Document conversion
- `pyyaml>=6.0` - Configuration management
- `rich>=13.0.0` - CLI formatting and output

**Data Processing:**
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `Pillow>=9.0.0` - Image processing

**Additional Tools:**
- `openai>=1.0.0` - AI-assisted content tools
- `gradio>=4.0.0` - Interactive interfaces
- `ghostscript>=0.7` - PDF compression
- `pre-commit>=3.0.0` - Code quality hooks

For the complete list, see `tools/dependencies/requirements.txt` and `pyproject.toml`.

---

## 🐧 Setting Things Up on **Linux**

### 1. 🔧 Install Quarto

Quarto is what drives the entire build process.

```sh
wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.31/quarto-1.7.31-linux-amd64.deb
sudo dpkg -i quarto-1.7.31-linux-amd64.deb
```

Test it with:

```sh
quarto --version
```

---

### 2. 📊 Install R

If you're using Ubuntu or Debian:

```sh
sudo apt-get update
sudo apt-get install -y r-base
```

Test R:

```sh
R --version
```

---

### 3. 📦 Install Required R Packages

Once R is installed, open it by typing `R`, then run:

```r
install.packages("remotes")
source("tools/dependencies/install_packages.R")
```

This installs everything the book needs to render code, plots, etc. The R package dependencies are centrally managed in `tools/dependencies/install_packages.R`.

---

### 4. ✒️ Install TinyTeX (LaTeX Distribution)

TinyTeX is a lightweight version of TeX Live, which Quarto uses to generate PDFs.

```sh
quarto install tinytex
```

Then add it to your shell:

```sh
echo 'export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH' >> ~/.bashrc
source ~/.bashrc
```

---

### 5. 🧰 Install Additional TeX Live Packages (for diagrams, fonts, etc.)

These give us broader LaTeX support:

```sh
sudo apt-get install -y texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra \
  texlive-pictures texlive-luatex
```

---

### 6. 🖼️ Install Inkscape

This is needed to convert `.svg` images into `.pdf` (especially for TikZ diagrams).

```sh
sudo add-apt-repository ppa:inkscape.dev/stable -y
sudo apt-get update
sudo apt-get install -y inkscape
```

Test with:

```sh
inkscape --version
```

---

### 7. 📉 Install Ghostscript (for compressing the final PDF)

```sh
sudo apt-get install -y ghostscript
```

---

### 8. 🐍 Install Python 3.9+ and Dependencies

```sh
sudo apt-get install -y python3 python3-pip python3-venv
```

Test with:

```sh
python3 --version    # Should be 3.9 or higher
pip3 --version
```

### 9. 📦 Install Python Dependencies

The project uses modern Python packaging. Install all dependencies with:

```sh
# Option 1: Using pip (recommended)
pip install -r requirements.txt

# Option 2: Install in development mode (includes CLI as command)
pip install -e .

# Option 3: Using a virtual environment (best practice)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**What gets installed:**
- Book Binder CLI and all build tools
- Jupyter and Quarto integration packages
- Data processing libraries (pandas, numpy)
- AI/ML tools for content assistance
- Pre-commit hooks for code quality

The `requirements.txt` file points to `tools/dependencies/requirements.txt`, which contains all production and development dependencies.

---

### 10. 🧪 Test That It All Works

Once you've installed everything, run the health check:

```sh
./binder doctor
```

This will verify:
- ✅ Quarto installation
- ✅ Python and dependencies
- ✅ R and required packages
- ✅ LaTeX and TinyTeX
- ✅ Inkscape and Ghostscript
- ✅ Configuration files
- ✅ Build directory structure

If everything passes, you're ready to build the book!

---

## 🧱 How to Build the Book

Navigate to the root folder of the project:

```sh
cd path/to/MLSysBook
```

### 🚀 **Dual-Configuration System**

The book uses a **dual-configuration approach** that automatically switches between optimized settings for different output formats:

- **`quarto/config/_quarto-html.yml`** → Optimized for interactive website (clean navigation, TikZ→SVG, cross-references)
- **`quarto/config/_quarto-pdf.yml`** → Optimized for academic PDF (full citations, LaTeX rendering, book structure)
- **`quarto/config/_quarto-epub.yml`** → Optimized for EPUB (e-reader format, reflowable content)

The Binder CLI automatically handles configuration switching using symlinks — **no manual file management needed!**

---

### 🔹 **Build Commands (Book Binder CLI)**

The **recommended way** to build the book is using the Book Binder CLI:

#### Build Complete Book
```sh
./binder build                  # Complete website (HTML)
./binder pdf                    # Complete book (PDF)
./binder epub                   # Complete e-book (EPUB)
```

#### Build Specific Chapter(s)
```sh
./binder build intro                    # Single chapter (HTML)
./binder build intro,ml_systems         # Multiple chapters (HTML)
./binder pdf intro                      # Single chapter (PDF, selective build)
```

#### Preview Mode (Live Reload)
```sh
./binder preview                        # Preview complete book
./binder preview intro                  # Preview specific chapter
./binder preview intro,ml_systems       # Preview multiple chapters
```

#### Management Commands
```sh
./binder clean                  # Clean build artifacts
./binder status                 # Show current status
./binder list                   # List all available chapters
./binder doctor                 # Run comprehensive health check
./binder help                   # Show all commands
```

**Output Locations:**
- **HTML:** `build/html/`
- **PDF:** `build/pdf/`
- **EPUB:** `build/epub/`

---

### 🔹 **Advanced: Direct Quarto Commands**

If you need direct control without the Binder CLI:

#### Website (HTML) version:
```sh
cd quarto
ln -sf config/_quarto-html.yml _quarto.yml
quarto render --to html
```

#### PDF version:
```sh
cd quarto
ln -sf config/_quarto-pdf.yml _quarto.yml
quarto render --to=titlepage-pdf
```

#### EPUB version:
```sh
cd quarto
ln -sf config/_quarto-epub.yml _quarto.yml
quarto render --to epub
```

**Important:** The Binder CLI is strongly recommended as it:
- ✅ Handles configuration switching automatically
- ✅ Manages build artifacts and cleanup
- ✅ Provides progress indicators
- ✅ Validates system health
- ✅ Supports fast/selective builds

---

## 🪟 Setup on **Windows**

### Prerequisites
- Windows 10 or later
- Administrator access for some installations

### 1. Install Quarto
Download and install from [quarto.org](https://quarto.org/docs/download/)

### 2. Install Python 3.9+
Download from [python.org](https://www.python.org/downloads/) or use Windows Store.

**Important:** Check "Add Python to PATH" during installation.

### 3. Install R
Download from [CRAN](https://cran.r-project.org/)

### 4. Install R Packages
Open R and run:
```r
install.packages("remotes")
source("tools/dependencies/install_packages.R")
```

### 5. Install TinyTeX
From R console:
```r
install.packages("tinytex")
tinytex::install_tinytex()
```

### 6. Install Inkscape, Ghostscript (Using Chocolatey)
Open PowerShell (as Administrator):
```powershell
# Install Chocolatey if not already installed
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install tools
choco install inkscape ghostscript -y
```

### 7. Install Python Dependencies
Open Command Prompt or PowerShell in the project directory:
```powershell
# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 8. Test Everything Works
Run the health check:
```powershell
python binder doctor
```

Or test building:
```powershell
python binder build intro
python binder pdf
```

---

## 💡 Troubleshooting Tips

### Common Installation Issues

**Quarto not found?**
```sh
# Verify installation
quarto --version

# Check PATH (Linux/macOS)
echo $PATH | grep quarto

# Reinstall if needed
# Linux: sudo dpkg -i quarto-*.deb
# macOS: brew install --cask quarto
# Windows: Download from quarto.org
```

**Python version issues?**
```sh
# Check Python version (must be 3.9+)
python --version
python3 --version

# Use specific version if multiple installed
python3.9 --version
```

**Dependencies not installing?**
```sh
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Try with verbose output
pip install -r requirements.txt -v

# If SSL errors occur
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Build Issues

**PDF build fails?**
- Verify LaTeX is installed: `pdflatex --version`
- Verify Inkscape is installed: `inkscape --version`
- Check TinyTeX path: `tinytex::tinytex_root()` in R
- Try rebuilding from scratch:
  ```sh
  ./binder clean
  ./binder pdf
  ```

**Chapter not found?**
```sh
# List all available chapters
./binder list

# Use exact chapter names (case-sensitive)
./binder build intro    # ✓ correct
./binder build Intro    # ✗ wrong
```

**Build artifacts detected?**
```sh
# Clean all build artifacts
./binder clean

# Check status
./binder status

# Run health check
./binder doctor
```

**Configuration issues?**
```sh
# Check current configuration
ls -la quarto/_quarto.yml

# Should be a symlink to config/_quarto-html.yml or config/_quarto-pdf.yml
# If not, recreate:
cd quarto
ln -sf config/_quarto-html.yml _quarto.yml
```

### System-Specific Issues

**macOS: Inkscape not in PATH?**
```sh
# Add Inkscape to PATH
echo 'export PATH="/Applications/Inkscape.app/Contents/MacOS:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Linux: Missing system libraries?**
```sh
# Install common missing libraries
sudo apt-get install -y libcairo2-dev libharfbuzz-dev libfribidi-dev \
  libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev
```

**Windows: Permission errors?**
```powershell
# Run PowerShell as Administrator
# Disable execution policy temporarily
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### Getting Help

If you're still having issues:

1. **Run the health check**: `./binder doctor`
2. **Check the logs**: Look for detailed error messages
3. **Consult documentation**:
   - [BINDER.md](BINDER.md) - Binder CLI guide
   - [DEVELOPMENT.md](DEVELOPMENT.md) - Development setup
4. **Ask for help**:
   - GitHub Discussions: https://github.com/harvard-edge/cs249r_book/discussions
   - GitHub Issues: https://github.com/harvard-edge/cs249r_book/issues

---

## 📦 Modern Python Packaging

The project uses modern Python packaging standards with `pyproject.toml`:

### Project Structure
```
MLSysBook/
├── pyproject.toml              # Python project configuration
├── requirements.txt            # Points to tools/dependencies/requirements.txt
├── tools/
│   └── dependencies/
│       ├── requirements.txt    # Actual dependencies
│       └── install_packages.R  # R dependencies
└── cli/                        # Modular CLI package
    ├── main.py                 # CLI entry point
    ├── commands/               # Command implementations
    ├── core/                   # Core functionality
    └── utils/                  # Utilities
```

### Installation Options

**Standard Installation (Recommended):**
```sh
pip install -r requirements.txt
```

**Development Installation:**
```sh
# Installs package in editable mode with CLI as command
pip install -e .

# Now you can use:
binder build
mlsysbook build  # Alternative command name
```

**With Optional Dependencies:**
```sh
# Install with AI features
pip install -e ".[ai]"

# Install with development tools
pip install -e ".[dev]"

# Install everything
pip install -e ".[ai,dev]"
```

### Key Features

The `pyproject.toml` defines:
- **Minimum Python version**: 3.9+
- **Core dependencies**: Listed in `dependencies` section
- **Optional dependencies**: AI tools, dev tools, build tools
- **Entry points**: `binder` and `mlsysbook` commands
- **Code quality tools**: Black, isort, pylint, mypy configurations
- **Testing setup**: Pytest with coverage

### Benefits
- ✅ Standards-compliant packaging
- ✅ Proper dependency management
- ✅ CLI installed as system command
- ✅ Supports pip, poetry, and other tools
- ✅ Easy distribution and installation

---

## 🎉 That's It!

Once everything is set up, you'll be able to:

### Development Workflow
- 🚀 **Preview changes locally** with live reload: `./binder preview intro`
- 🔨 **Build individual chapters** for fast iteration: `./binder build intro`
- 📚 **Build complete book** in multiple formats: `./binder build`, `./binder pdf`, `./binder epub`
- 🔍 **Validate your setup** anytime: `./binder doctor`
- 🧹 **Clean up artifacts**: `./binder clean`

### Contributing
- 📝 **Make edits** to chapter content in `quarto/contents/`
- ✅ **Test locally** before committing
- 🤝 **Follow best practices** with pre-commit hooks
- 💪 **Contribute like a pro** to the open-source book

### Next Steps
1. Read [BINDER.md](BINDER.md) for complete CLI reference
2. Check [DEVELOPMENT.md](DEVELOPMENT.md) for development workflow
3. Review [contribute.md](contribute.md) for contribution guidelines
4. Join discussions at [GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions)

---

## 📖 Additional Resources

### Documentation
- **[BINDER.md](BINDER.md)** - Complete Book Binder CLI reference
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development guidelines and workflow
- **[contribute.md](contribute.md)** - Contribution guidelines
- **[PUBLISH_LIVE_WORKFLOW.md](PUBLISH_LIVE_WORKFLOW.md)** - Publishing workflow

### Community
- **[GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions)** - Ask questions and share knowledge
- **[GitHub Issues](https://github.com/harvard-edge/cs249r_book/issues)** - Report bugs and request features
- **[MLSysBook.ai](https://mlsysbook.ai)** - Main website and learning platform

### Tools and Scripts
The `tools/scripts/` directory contains various utilities:
- **`content/`** - Content management tools
- **`cross_refs/`** - Cross-reference management
- **`genai/`** - AI-assisted content tools
- **`glossary/`** - Glossary management
- **`maintenance/`** - System maintenance scripts
- **`publish/`** - Publishing and deployment tools

Run `./binder help` to see all available commands and their descriptions.

---

## 🙏 Contributing

We welcome contributions! The easiest way to get started:

1. **Fork and clone** the repository
2. **Set up your environment**: `./binder setup`
3. **Make your changes** to content or code
4. **Test locally**: `./binder preview <chapter>`
5. **Submit a pull request**

For detailed contribution guidelines, see [contribute.md](contribute.md).

---

**Last Updated**: October 2025
**Project**: Machine Learning Systems - Principles and Practices
**Website**: https://mlsysbook.ai
