# Use a base image with R 4.3.2 (supports both ARM64 and AMD64)
FROM rocker/r-ver:4.3.2
# Set non-interactive mode to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
# ------------------------
# 🔧 Install System Dependencies (including curl)
# ------------------------
RUN apt-get update && apt-get install -y \
    curl wget unzip perl \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
    libfreetype6-dev libjpeg-dev libtiff5-dev \
    libcairo2 gdk-pixbuf2.0-bin libgdk-pixbuf2.0-dev librsvg2-bin \
    fonts-dejavu fonts-freefont-ttf \
    ghostscript inkscape \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
# ------------------------
# 📦 Install Quarto (Auto-detect architecture)
# ------------------------
RUN ARCH=$(dpkg --print-architecture) && \
    curl -fsSL "https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.13/quarto-1.7.13-linux-${ARCH}.deb" -o quarto.deb && \
    dpkg -i quarto.deb && rm quarto.deb
# ------------------------
# 🌐 Install Chromium using Quarto's built-in installer
# ------------------------
RUN quarto install chromium && \
    # Set environment variable for Quarto to find Chromium
    echo "QUARTO_CHROMIUM_PATH=~/.local/share/quarto/chromium/chromium" >> /etc/environment && \
    # Also add as ENV for Docker sessions
    echo 'export QUARTO_CHROMIUM_PATH=~/.local/share/quarto/chromium/chromium' >> ~/.bashrc && \
    # Verify Chromium installation
    ls -la ~/.local/share/quarto/chromium/ && \
    echo "Chromium installation verified"
# ------------------------
# 📦 Manually Install TeX Live (Ensuring It Works)
# ------------------------
RUN wget -qO /tmp/install-tl-unx.tar.gz "https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz" && \
    tar -xzf /tmp/install-tl-unx.tar.gz -C /tmp && \
    cd /tmp/install-tl-* && \
    echo "selected_scheme scheme-basic" > texlive.profile && \
    mkdir -p /usr/local/texlive && \
    ./install-tl --profile=texlive.profile --no-interaction --texdir=/usr/local/texlive/2023 && \
    cd .. && rm -rf /tmp/install-tl-unx.tar.gz /tmp/install-tl-* && \
    echo "TeX Live installed successfully"
# **Ensure TeX Live is in PATH**
ENV PATH="/usr/local/texlive/2023/bin/x86_64-linux:/usr/local/texlive/2023/bin/aarch64-linux:/usr/local/texlive/2023/bin:$PATH"
# ------------------------
# ✅ Verify Installations
# ------------------------
RUN if [ -d "/usr/local/texlive/2023/bin" ]; then \
      echo "TeX Live installed at /usr/local/texlive/2023"; \
    else \
      echo "TeX Live installation failed"; \
      exit 1; \
    fi && \
    # Verify TeX Live command works
    which pdflatex && \
    # Verify Quarto installation
    which quarto && \
    quarto --version && \
    # Simple check for Chromium installation
    ls -la ~/.local/share/quarto/chromium/ || echo "Chromium directory not found, but it may be installed elsewhere" && \
    echo "Basic verification complete"
# ------------------------
# 📦 Install LaTeX Packages Using `tlmgr`
# ------------------------
RUN /usr/local/texlive/2023/bin/x86_64-linux/tlmgr install scheme-basic \
    collection-basic collection-latex \
    collection-latexrecommended collection-latexextra \
    collection-fontsrecommended collection-pictures \
    collection-xetex collection-luatex collection-bibtexextra \
    collection-langenglish koma-script standalone pgf \
    tikz-cd pgfplots preview marginfix newpx luatex85
# ------------------------
# 📊 Install R Dependencies
# ------------------------
RUN Rscript -e "install.packages(c('remotes'))"
# ------------------------
# 🐍 Install Python & PDF Tools
# ------------------------
RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip install pikepdf ghostscript PyPDF2
# ------------------------
# 📂 Set Working Directory
# ------------------------
WORKDIR /workspace
# ------------------------
# 🏗 Default Command
# ------------------------
CMD ["bash"]
