# Quarto Build Guide

## Overview

This guide provides step-by-step instructions for building a Quarto project locally on Linux and Windows. It covers installation of dependencies, rendering Quarto projects, and troubleshooting potential issues.

## Prerequisites

### Required Software

Ensure that you have the following installed:

- **Quarto** (Version 1.7.13 recommended, but can be updated as needed)
- **R** (Version 4.3.2 recommended, if using R-based Quarto features)
- **TinyTeX** (for PDF builds)
- **TeX Live** (for full LaTeX support)
- **Inkscape** (for SVG to PDF conversions, if required)
- **Ghostscript** (for PDF compression, if applicable)
- **Python 3** (for PDF compression tools, if applicable)
- **System Dependencies** (libpango, libfontconfig, and others for Linux builds)

## Installation Instructions

### Linux Setup

1. **Install Quarto:**
   ```bash
   wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.13/quarto-1.7.13-linux-amd64.deb
   sudo dpkg -i quarto-1.7.13-linux-amd64.deb
   ```
   
2. **Install TinyTeX:**
   ```bash
   quarto install tinytex
   ```
   Add TinyTeX to PATH:
   ```bash
   echo "export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH" >> ~/.bashrc
   source ~/.bashrc
   ```
   
3. **Install Required TeX Live Packages:**
   ```bash
   sudo apt-get update && sudo apt-get install -y \
       texlive texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra \
       texlive-lang-english texlive-pictures texlive-xetex texlive-luatex
   ```
   
4. **Install System Dependencies for Linux:**
   ```bash
   sudo apt-get install -y \
       libpangoft2-1.0-0 fonts-dejavu fonts-freefont-ttf libpango-1.0-0 \
       libpangocairo-1.0-0 libcogl-pango-dev pango1.0-tools libcairo2 \
       gdk-pixbuf2.0-bin libgdk-pixbuf2.0-dev librsvg2-bin libcurl4-openssl-dev \
       libssl-dev libxml2-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
       libfreetype6-dev libtiff5-dev libjpeg-dev
   ```
   
5. **Install Inkscape (if required for graphics processing):**
   ```bash
   sudo add-apt-repository ppa:inkscape.dev/stable -y
   sudo apt-get update
   sudo apt-get install inkscape -y
   ```
   
6. **Install Ghostscript (for PDF compression, optional):**
   ```bash
   sudo apt-get install ghostscript -y
   ```
   
7. **Install Python 3 (for PDF compression and other utilities):**
   ```bash
   sudo apt-get install python3 python3-pip -y
   ```

8. **Install R Packages:**
   ```r
   install.packages(c("remotes"))
   source("install_packages.R")
   ```

### Windows Setup

1. **Install Quarto:**
   - Download the installer from [Quarto’s website](https://quarto.org/docs/download/)
   - Run the installer and follow the prompts.

2. **Install TinyTeX:**
   - Open R and run:
     ```r
     install.packages("tinytex")
     tinytex::install_tinytex()
     ```

3. **Install TeX Live (if additional LaTeX support is needed):**
   - Download the TeX Live installer from [TUG.org](https://www.tug.org/texlive/)
   - Follow the installation prompts.

4. **Install Inkscape (if required for graphics processing):**
   ```powershell
   choco install inkscape -y
   ```
   
5. **Install Ghostscript (for PDF compression, optional):**
   ```powershell
   choco install ghostscript -y
   ```
   
6. **Install Python 3 (for PDF compression tools, if applicable):**
   ```powershell
   choco install python -y
   ```

7. **Install R Packages:**
   ```r
   install.packages(c("remotes"))
   source("install_packages.R")
   ```

## Building the Quarto Project

1. **Navigate to the Quarto Project Directory:**
   ```bash
   cd path/to/your/project
   ```

2. **Render the project to HTML:**
   ```bash
   quarto render --to html
   ```
   
3. **Render the project to PDF:**
   ```bash
   quarto render --to pdf
   ```
   If errors occur related to missing LaTeX packages, install them using TeX Live.

4. **Compress PDF (Linux only, if required):**
   ```bash
   python3 ./scripts/quarto_publish/gs_compress_pdf.py \
       -i ./_book/Machine-Learning-Systems.pdf \
       -o ./_book/ebook.pdf \
       -s "/ebook"
   ```

## Debugging Common Issues

### Quarto Not Found

Ensure that Quarto is installed and available in your PATH:
```bash
quarto --version
```

### PDF Compilation Errors

- Ensure TinyTeX and necessary LaTeX packages are installed.
- Run `tlmgr install <missing-package>` if a package is missing.

### Fonts or Graphics Not Rendering

- Ensure Inkscape is installed and accessible.
- Check if required fonts are installed on your system.

### PDF Compression Script Fails

- Ensure Python and Ghostscript are installed.
- Check for missing dependencies using `pip install -r requirements.txt` if applicable.

## Conclusion

Following these steps will allow you to build Quarto projects efficiently on both Linux and Windows. If issues persist, refer to the [Quarto documentation](https://quarto.org/docs/) for further troubleshooting.

