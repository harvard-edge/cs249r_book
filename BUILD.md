# **📚 Quarto Build Guide**

## **🔹 Overview**
This guide provides step-by-step instructions for **building a Quarto project** on **Linux and Windows**.  
It covers **two methods**:
1. **Using Docker (Recommended)** ✅ → No manual installation needed.
2. **Manual Installation (Alternative)** 🛠 → If you cannot use Docker.

---

## **1️⃣ 🚀 Using Docker (Recommended)**
If you have **Docker installed**, you **don’t need to manually install Quarto, TeX Live, R, or any dependencies**.  
Everything is **pre-configured** in the Docker image.

### **📥 Prerequisites**
- Install **Docker** ([Download Here](https://docs.docker.com/get-docker/))

### **🔨 Steps to Build Your Quarto Project**
1. **Navigate to your project folder**  
   ```sh
   cd path/to/your/project
   ```

2. **Render the Quarto project using Docker**  
   ```sh
   docker run --rm -v "$(pwd):/workspace" -w /workspace profvjreddi/quarto-build quarto render
   ```

3. **(Optional) Compress the PDF Output**  
   ```sh
   docker run --rm -v "$(pwd):/workspace" -w /workspace profvjreddi/quarto-build python3 ./scripts/quarto_publish/gs_compress_pdf.py -i ./_book/Machine-Learning-Systems.pdf -o ./_book/ebook.pdf -s "/ebook"
   ```

4. **(Optional) Enter the Docker Container for Debugging**  
   ```sh
   docker run --rm -it -v "$(pwd):/workspace" -w /workspace profvjreddi/quarto-build bash
   ```
   Inside the container, test:
   ```sh
   quarto --version
   pdflatex --version
   tlmgr --version
   ```

---

## **2️⃣ 🛠 Manual Installation (Alternative)**
If you **cannot use Docker**, follow these steps to manually **install all required dependencies**.

### **📥 Prerequisites**
You will need:
- **Quarto**
- **R** (if using R-based Quarto features)
- **TinyTeX & TeX Live** (for PDF builds)
- **Inkscape** (for SVG to PDF conversions, if needed)
- **Ghostscript** (for PDF compression, optional)
- **Python 3** (for PDF compression tools, optional)
- **System Dependencies** (various libraries for Linux)

---

### **📌 Linux Setup**
1. **Install Quarto**
   ```sh
   wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.13/quarto-1.7.13-linux-amd64.deb
   sudo dpkg -i quarto-1.7.13-linux-amd64.deb
   ```

2. **Install TinyTeX**
   ```sh
   quarto install tinytex
   ```
   Add TinyTeX to PATH:
   ```sh
   echo "export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH" >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Install TeX Live**
   ```sh
   sudo apt-get update && sudo apt-get install -y texlive-full
   ```

4. **Install System Dependencies**
   ```sh
   sudo apt-get install -y libpangoft2-1.0-0 fonts-dejavu fonts-freefont-ttf \
       libpango-1.0-0 libpangocairo-1.0-0 libcogl-pango-dev pango1.0-tools \
       libcairo2 gdk-pixbuf2.0-bin libgdk-pixbuf2.0-dev librsvg2-bin \
       libcurl4-openssl-dev libssl-dev libxml2-dev libfontconfig1-dev \
       libharfbuzz-dev libfribidi-dev libfreetype6-dev libtiff5-dev libjpeg-dev
   ```

5. **Install Inkscape (if required for graphics processing)**
   ```sh
   sudo add-apt-repository ppa:inkscape.dev/stable -y
   sudo apt-get update
   sudo apt-get install inkscape -y
   ```

6. **Install Ghostscript (for PDF compression, optional)**
   ```sh
   sudo apt-get install ghostscript -y
   ```

7. **Install Python 3 (for PDF compression and utilities)**
   ```sh
   sudo apt-get install python3 python3-pip -y
   ```

8. **Install R Packages (if using R in Quarto)**
   ```r
   install.packages(c("remotes"))
   source("install_packages.R")
   ```

---

### **📌 Windows Setup**
1. **Install Quarto**  
   - Download from [Quarto’s website](https://quarto.org/docs/download/)
   - Run the installer.

2. **Install TinyTeX**  
   - Open R and run:
     ```r
     install.packages("tinytex")
     tinytex::install_tinytex()
     ```

3. **Install TeX Live (if additional LaTeX support is needed)**  
   - Download the TeX Live installer from [TUG.org](https://www.tug.org/texlive/)
   - Follow the installation prompts.

4. **Install Inkscape (if required for graphics processing)**  
   ```powershell
   choco install inkscape -y
   ```

5. **Install Ghostscript (for PDF compression, optional)**  
   ```powershell
   choco install ghostscript -y
   ```

6. **Install Python 3 (for PDF compression tools, if applicable)**  
   ```powershell
   choco install python -y
   ```

7. **Install R Packages (if using R in Quarto)**  
   ```r
   install.packages(c("remotes"))
   source("install_packages.R")
   ```

---

## **3️⃣ 🏗 Building the Quarto Project**
### **🚀 With Docker (Recommended)**
1. **Navigate to your project directory**
   ```sh
   cd path/to/your/project
   ```
2. **Run Quarto render**
   ```sh
   docker run --rm -v "$(pwd):/workspace" -w /workspace profvjreddi/quarto-build quarto render
   ```
3. **(Optional) Compress PDF**
   ```sh
   docker run --rm -v "$(pwd):/workspace" -w /workspace profvjreddi/quarto-build python3 ./scripts/quarto_publish/gs_compress_pdf.py -i ./_book/Machine-Learning-Systems.pdf -o ./_book/ebook.pdf -s "/ebook"
   ```

### **🛠 Manually (If Not Using Docker)**
1. **Navigate to the project directory**
   ```sh
   cd path/to/your/project
   ```
2. **Render the project**
   ```sh
   quarto render
   ```
3. **Compress PDF (Linux Only)**
   ```sh
   python3 ./scripts/quarto_publish/gs_compress_pdf.py -i ./_book/Machine-Learning-Systems.pdf -o ./_book/ebook.pdf -s "/ebook"
   ```

---

## **4️⃣ 🔍 Troubleshooting Common Issues**
### **Quarto Not Found**
```sh
quarto --version
```
If missing, install it manually or use Docker.

### **PDF Compilation Errors**
If `quarto render` fails due to missing LaTeX packages:
```sh
tlmgr install <missing-package>
```

### **Fonts or Graphics Not Rendering**
Ensure **Inkscape** is installed:
```sh
sudo apt install inkscape
```

### **PDF Compression Script Fails**
Ensure Python and Ghostscript are installed:
```sh
pip install pikepdf ghostscript PyPDF2
```

---

## **🎯 Conclusion**
🚀 **Using Docker is the easiest method.**  
```sh
docker run --rm -v "$(pwd):/workspace" -w /workspace profvjreddi/quarto-build quarto render
```
🛠 **Manual setup is still an option if needed.** 🤗
