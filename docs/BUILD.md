# 🛠 How to Build the Book Locally

Welcome! 👋 If you’re here, you’re probably trying to **build the Machine Learning Systems book locally** on your own machine.

This guide will walk you through **how to get set up manually**, especially if you're not using GitHub Actions or Docker. We'll cover what tools you need, why you need them, and how to test everything is working.

## 🚀 Quick Start (Recommended)

For most users, the easiest way is using our **Book Binder CLI**:

```bash
# First time setup
./binder setup

# Welcome and overview
./binder hello

# Quick preview
./binder preview intro

# Build everything
./binder build - html
./binder build - pdf
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

## ✅ What You’ll Need (And Why)

| Tool | Why It's Needed |
|------|------------------|
| **Quarto** | The core tool that converts the `.qmd` files into HTML/PDF |
| **R** | Some chapters include R code chunks and R-based plots |
| **R packages** | Supporting packages (defined in `install_packages.R`) |
| **TinyTeX + TeX Live** | Needed for LaTeX → PDF rendering |
| **Inkscape** | Converts `.svg` diagrams into `.pdf` (especially TikZ) |
| **Ghostscript** | Compresses large PDF files |
| **Python 3** | Needed for PDF compression scripts |
| **System libraries** | Fonts and rendering support on Linux systems |

Don’t worry — this guide will walk you through installing all of them, step by step.

---

## 🐧 Setting Things Up on **Linux**

### 1. 🔧 Install Quarto

Quarto is what drives the entire build process.

```sh
wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.13/quarto-1.7.13-linux-amd64.deb
sudo dpkg -i quarto-1.7.13-linux-amd64.deb
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
source("install_packages.R")
```

This installs everything the book needs to render code, plots, etc.

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

### 8. 🐍 Install Python 3 and pip (used for helper scripts)

```sh
sudo apt-get install -y python3 python3-pip
```

Test with:

```sh
python3 --version
pip3 --version
```

---

### 9. 🧪 Test That It All Works

Once you’ve installed everything, you're ready to try building the book!

---

## 🧱 How to Build the Book

Navigate to the root folder of the project:

```sh
cd path/to/MLSysBook
```

### 🚀 **NEW: Dual-Configuration System**

The book now uses a **dual-configuration approach** that automatically switches between optimized settings for different output formats:

- **`book/_quarto-html.yml`** → Optimized for interactive website (clean navigation, TikZ→SVG, no citations)
- **`book/_quarto-pdf.yml`** → Optimized for academic PDF (full citations, LaTeX rendering, book structure)

The build system automatically handles configuration switching using symlinks — **no manual file copying needed!**

---

### 🔹 **Build Commands (Recommended)**

Use these **automated commands** that handle configuration switching:

#### Interactive Build (Recommended)
```sh
make build
```
- Choose format interactively (HTML/PDF/Both)
- User-friendly prompts
- Perfect for development workflow

#### Build Website (HTML)
```sh
make build-html
```
- Uses HTML-optimized configuration
- TikZ diagrams → SVG conversion
- Clean navigation without chapter numbers
- Interactive quizzes and cross-references

#### Build PDF Book
```sh
make build-pdf  
```
- Uses PDF-optimized configuration  
- Full LaTeX rendering with citations
- Professional book formatting
- Traditional chapter numbering

#### Build Both Formats
```sh
make build-all
```

#### Development Preview
```sh
make preview        # HTML preview with live reload
make preview-pdf    # PDF preview
```

You'll find outputs in the `build/html/` folder for HTML and `build/pdf/` for PDF.

---

### 🔹 **Manual Commands (Advanced)**

If you need direct control, these commands work but require manual configuration management:

#### Website (HTML) version:
```sh
cd book
ln -sf _quarto-html.yml _quarto.yml
quarto render --to html
rm _quarto.yml
```

#### PDF version:
```sh
cd book  
ln -sf _quarto-pdf.yml _quarto.yml
quarto render --to titlepage-pdf
rm _quarto.yml
```

**Note:** The automated `make` commands are recommended as they handle configuration switching and cleanup automatically.

---

## 🪟 Setup on **Windows**

1. **Install Quarto**  
   Download from [quarto.org](https://quarto.org/docs/download/)

2. **Install R**  
   Download from [CRAN](https://cran.r-project.org/)

3. **Install R Packages**  
   Open R and run:
   ```r
   install.packages("remotes")
   source("install_packages.R")
   ```

4. **Install TinyTeX**  
   ```r
   install.packages("tinytex")
   tinytex::install_tinytex()
   ```

5. **Install Inkscape, Ghostscript, Python**  
   Open PowerShell (as Administrator), then run:
   ```powershell
   choco install inkscape ghostscript python -y
   ```

6. **Test Everything Works**  
   Open a new terminal and try:
   ```powershell
   quarto render --to html
   quarto render --to titlepage-pdf
   ```

---

## 💡 Troubleshooting Tips

**Quarto not found?**  
Make sure it’s in your PATH and installed correctly.

**PDF build fails?**  
- Check that LaTeX and Inkscape are working.
- Make sure you're using `--to titlepage-pdf` and not just `--to pdf`.

**Compression script doesn’t work?**  
- Make sure Ghostscript is installed and accessible.
- You may need to install Python packages:
  ```sh
  pip3 install pikepdf ghostscript PyPDF2
  ```

---

## 🎉 That’s It!

Once everything is set up, you’ll be able to:

- Preview changes locally
- Build clean HTML and PDF versions
- Contribute to the book like a pro 💪

Let me know if you'd like this saved as `manual_setup.md` or included in your Quarto documentation!

---

## 🔧 Additional Troubleshooting

**Icon files missing for foldbox callouts?**  
If you see errors like `File 'icon_callout-quiz-question.pdf' not found`, the PNG icons need to be converted to PDF format for LaTeX rendering:

```sh
cd book/_extensions/ute/custom-numbered-blocks/style/icons
convert icon_callout-quiz-question.png icon_callout-quiz-question.pdf
convert icon_callout-quiz-answer.png icon_callout-quiz-answer.pdf
convert icon_callout-chapter-connection.png icon_callout-chapter-connection.pdf
convert icon_callout-resource-exercises.png icon_callout-resource-exercises.pdf
convert Icon_callout-resource-slides.png icon_callout-resource-slides.pdf
convert Icon_callout-resource-videos.png icon_callout-resource-videos.pdf
```

**Note:** This requires ImageMagick to be installed. On macOS: `brew install imagemagick`, on Ubuntu: `sudo apt-get install imagemagick`.
