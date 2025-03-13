# 🛠 How to Build the Book Locally

Welcome! 👋 If you’re here, you’re probably trying to **build the Machine Learning Systems book locally** on your own machine.

This guide will walk you through **how to get set up manually**, especially if you're not using GitHub Actions or Docker. We'll cover what tools you need, why you need them, and how to test everything is working.

---

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
cd path/to/the/book
```

### 🔹 To Build the **Website (HTML)** version:
```sh
quarto render --to html
```

You’ll find the output in the `_book/` folder — you can open `index.html` in your browser to preview it.

---

### 🔹 To Build the **PDF** version:
```sh
quarto render --to titlepage-pdf
```

This uses a custom format defined in `_quarto.yml`. The output will be in:

```
_book/Machine-Learning-Systems.pdf
```

---

### 🔹 (Optional) Compress the PDF
The final PDF can be large. You can compress it using the provided script:

```sh
python3 ./scripts/quarto_publish/gs_compress_pdf.py \
  -i ./_book/Machine-Learning-Systems.pdf \
  -o ./_book/ebook.pdf \
  -s "/ebook"
```

If you're happy with the result, replace the original:

```sh
mv ./_book/ebook.pdf ./_book/Machine-Learning-Systems.pdf
```

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
