# 🛠 How to Build the Book Locally

Welcome! 👋 If you're here, you're probably trying to **build the Machine Learning Systems book locally** on your own machine.

This guide will walk you through **how to get set up manually** for local development. The instructions are based on our production GitHub Actions workflow but adapted for local use.

---

## 📚 What Are We Building?

This project uses [**Quarto**](https://quarto.org) to render multiple formats:

- **HTML**: Website version of the book (default)
- **PDF**: Typeset PDF using custom titlepage format
- **EPUB**: E-book format

The HTML version builds easily, but **PDF and EPUB require additional dependencies** like LaTeX, Inkscape, and specialized packages.

---

## ✅ Prerequisites Overview

| Tool | Version | Purpose |
|------|---------|----------|
| **Quarto** | 1.7.31+ | Core rendering engine |
| **Python** | 3.13+ | Build tools and compression scripts |
| **R** | 4.3.2+ | Code execution and plotting |
| **TeX Live** | 2025 | PDF typesetting and LaTeX packages |
| **Inkscape** | Latest | SVG to PDF conversion for diagrams |
| **Ghostscript** | Latest | PDF compression and optimization |
| **System Libraries** | Various | Font rendering and graphics support |

> **Note**: These versions match our production GitHub Actions workflow for consistency.

---

## 🐧 Linux Setup (Ubuntu/Debian)

### 1. 🔧 Install Quarto

```bash
# Download and install Quarto (matches production version)
wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.31/quarto-1.7.31-linux-amd64.deb
sudo dpkg -i quarto-1.7.31-linux-amd64.deb

# Verify installation
quarto --version
quarto check
```

### 2. 🐍 Install Python and Dependencies

```bash
# Install Python 3.13+ (or latest available)
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

# Install Python dependencies
pip3 install -r requirements.txt

# Verify installation
python3 --version
```

### 3. 📊 Install R and Packages

```bash
# Install R 4.3.2+ (or latest)
sudo apt-get install -y r-base

# Install R packages using the provided script
Rscript install_packages.R

# Verify R installation
R --version
```

### 4. 📚 Install TeX Live (Production Method)

We use the same TeX Live setup as our GitHub Actions:

```bash
# Install TeX Live base
wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh

# Add to PATH
echo 'export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Install required TeX packages (matches tl_packages file)
tlmgr install \
  collection-basic \
  collection-fontsextra \
  collection-fontutils \
  collection-latex \
  collection-latexextra \
  collection-latexrecommended \
  collection-fontsrecommended \
  collection-luatex \
  collection-pictures
```

### 5. 🖼️ Install Inkscape (Production Method)

```bash
# Remove any existing Inkscape
sudo apt-get remove -y inkscape || true

# Install from PPA for reliable version
sudo add-apt-repository ppa:inkscape.dev/stable -y
sudo apt-get update
sudo apt-get install -y inkscape

# Install font dependencies
sudo apt-get install -y \
  fonts-freefont-ttf \
  fonts-liberation \
  fontconfig

# Update font cache
sudo fc-cache -fv

# Test Inkscape
inkscape --version
```

### 6. 📉 Install Ghostscript

```bash
sudo apt-get install -y ghostscript

# Verify installation
gs --version
```

### 7. 🛠️ Install System Dependencies

```bash
# Install system libraries (matches GitHub Actions)
sudo apt-get install -y \
  fonts-dejavu \
  fonts-freefont-ttf \
  gdk-pixbuf2.0-bin \
  libcairo2 \
  libfontconfig1 \
  libfreetype6 \
  libpango-1.0-0 \
  libpangocairo-1.0-0 \
  libpangoft2-1.0-0 \
  libxml2-dev \
  libcurl4-openssl-dev \
  libjpeg-dev \
  libtiff5-dev \
  libpng-dev
```

### 8. ✅ Verify Installation

```bash
# Test all components
echo "Testing Quarto..."
quarto check

echo "Testing Python dependencies..."
python3 -c "import nltk, openai, gradio; print('Python deps OK')"

echo "Testing R packages..."
Rscript -e "library(ggplot2); library(knitr); cat('R packages OK\\n')"

echo "Testing Inkscape..."
echo '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><circle cx="50" cy="50" r="40" fill="red"/></svg>' > test.svg
inkscape --export-type=pdf --export-filename=test.pdf test.svg && echo "Inkscape OK" && rm test.svg test.pdf

echo "Testing Ghostscript..."
gs --version > /dev/null && echo "Ghostscript OK"

echo "✅ All dependencies verified!"
```

---

## 🧱 Building the Book

Navigate to the project root and use these commands:

### 🌐 Build HTML (Website)
```bash
# Build website version
quarto render --to html

# View output
open _book/index.html  # macOS
# or
xdg-open _book/index.html  # Linux
```

### 📖 Build PDF (Typeset)
```bash
# Build PDF with custom titlepage format
quarto render --to titlepage-pdf

# Output location
ls -lh _book/Machine-Learning-Systems.pdf
```

### 📱 Build EPUB (E-book)
```bash
# Build EPUB format
quarto render --to epub

# Output location
ls -lh _book/Machine-Learning-Systems.epub
```

### 🚀 Build All Formats
```bash
# Build everything (HTML, PDF, EPUB)
quarto render
```

### 📉 Compress PDF (Production Method)

The PDF can be large. Use Ghostscript compression (matches production):

```bash
# Compress PDF using production settings
gs \
  -sDEVICE=pdfwrite \
  -dCompatibilityLevel=1.4 \
  -dPDFSETTINGS=/ebook \
  -dNOPAUSE \
  -dQUIET \
  -dBATCH \
  -sOutputFile="_book/ebook.pdf" \
  "_book/Machine-Learning-Systems.pdf"

# Replace original with compressed version
mv _book/ebook.pdf _book/Machine-Learning-Systems.pdf

# Check file size
ls -lh _book/Machine-Learning-Systems.pdf
```

---

## 🪟 Windows Setup

### Option 1: Using Chocolatey (Recommended)

```powershell
# Install Chocolatey if not already installed
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install all dependencies
choco install quarto python r.project inkscape ghostscript -y

# Install Python dependencies
pip install -r requirements.txt

# Install R packages
Rscript install_packages.R
```

### Option 2: Manual Installation

1. **Install Quarto 1.7.31+**  
   Download from [quarto.org](https://quarto.org/docs/download/)

2. **Install Python 3.13+**  
   Download from [python.org](https://www.python.org/downloads/)
   ```powershell
   pip install -r requirements.txt
   ```

3. **Install R 4.3.2+**  
   Download from [CRAN](https://cran.r-project.org/)
   ```r
   source("install_packages.R")
   ```

4. **Install Inkscape**  
   Download from [inkscape.org](https://inkscape.org/release/)

5. **Install Ghostscript**  
   Download from [ghostscript.com](https://www.ghostscript.com/download/gsdnld.html)

### Windows Build Commands

```powershell
# Build HTML
quarto render --to html

# Build PDF
quarto render --to titlepage-pdf

# Compress PDF (Windows method)
$input = "./_book/Machine-Learning-Systems.pdf"
$output = "./_book/ebook.pdf"
gswin64c -sDEVICE=pdfwrite -dCompatibilityLevel:1.4 -dPDFSETTINGS=/ebook -dNOPAUSE -dBATCH -sOutputFile="$output" "$input"
Move-Item -Force $output $input
```

---

## 🔧 Development Workflow

### Quick Development Loop
```bash
# 1. Make changes to .qmd files
# 2. Preview changes (HTML is fastest)
quarto render --to html

# 3. Test locally
open _book/index.html

# 4. Build PDF when ready
quarto render --to titlepage-pdf
```

### Using Quarto Preview
```bash
# Live preview with auto-reload (HTML only)
quarto preview

# Preview will be available at http://localhost:4200
```

---

## 🐛 Troubleshooting

### Common Issues

**❌ Quarto not found**  
```bash
# Check installation
quarto --version
which quarto  # Linux/Mac
where quarto  # Windows
```

**❌ PDF build fails**  
```bash
# Check LaTeX installation
tlmgr --version

# Check Inkscape
inkscape --version

# Ensure using correct format
quarto render --to titlepage-pdf  # ✅ Correct
quarto render --to pdf            # ❌ Wrong
```

**❌ Missing LaTeX packages**  
```bash
# Install missing packages
tlmgr install <package-name>

# Or reinstall collections
tlmgr install collection-latex collection-pictures
```

**❌ Inkscape SVG conversion fails**  
```bash
# Test Inkscape directly
echo '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100" height="100" fill="blue"/></svg>' > test.svg
inkscape --export-type=pdf --export-filename=test.pdf test.svg
```

**❌ Python dependencies missing**  
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade

# Check specific imports
python3 -c "import nltk, openai, gradio"
```

### Performance Tips

- **Use HTML for development**: Much faster than PDF
- **Cache heavy operations**: R packages and TeX Live installations
- **Parallel builds**: Use `quarto render --parallel` if available
- **Incremental builds**: Only changed files will be re-rendered

### Getting Help

1. **Check GitHub Actions**: Our `.github/workflows/quarto-build.yml` shows the exact production setup
2. **Compare versions**: Ensure your local versions match the workflow
3. **Test individual components**: Verify each tool works independently
4. **Check logs**: Quarto provides detailed error messages

---

## 🎯 Production Parity

These instructions match our GitHub Actions workflow:
- **Quarto**: 1.7.31
- **Python**: 3.13+
- **R**: 4.3.2+
- **TeX Live**: 2025 collections
- **Inkscape**: Latest stable
- **Ghostscript**: Latest stable

For exact package versions, see:
- `requirements.txt` (Python)
- `install_packages.R` (R)
- `tl_packages` (TeX Live)
- `.github/workflows/quarto-build.yml` (Full workflow)