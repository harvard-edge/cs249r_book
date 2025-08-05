# MLSysBook Quarto Build Container
# Based on Ubuntu 22.04 with all dependencies pre-installed
# This container eliminates the 30-45 minute setup time for Linux builds

FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV R_LIBS_USER=/usr/local/lib/R/library
ENV QUARTO_LOG_LEVEL=INFO
ENV PYTHONIOENCODING=utf-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV PATH=/usr/local/texlive/bin/x86_64-linux:$PATH

# === PHASE 1: SYSTEM DEPENDENCIES ===
RUN echo "🚀 === STARTING SYSTEM DEPENDENCIES INSTALLATION ===" && \
    echo "⏰ Estimated time: 2-3 minutes" && \
    echo "📊 Free disk space: $(df -h / | tail -1 | awk '{print $4}')" && \
    start_time=$(date +%s) && \
    \
    echo "🔄 Updating package lists..." && \
    apt-get update && \
    \
    echo "📦 Installing core system packages (25 packages)..." && \
    apt-get install -y \
    fonts-dejavu \
    fonts-freefont-ttf \
    gdk-pixbuf2.0-bin \
    libcairo2 \
    libfontconfig1 \
    libfontconfig1-dev \
    libfreetype6 \
    libfreetype6-dev \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libxml2-dev \
    libcurl4-openssl-dev \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    librsvg2-dev \
    libgdal-dev \
    libudunits2-dev \
    wget \
    curl \
    git && \
    \
    echo "🧹 Cleaning package cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    \
    end_time=$(date +%s) && \
    duration=$((end_time - start_time)) && \
    echo "✅ === SYSTEM DEPENDENCIES COMPLETE === (${duration}s)" && \
    echo "📊 Free disk space: $(df -h / | tail -1 | awk '{print $4}')"

# === PHASE 2: INKSCAPE INSTALLATION ===
RUN echo "🚀 === STARTING INKSCAPE INSTALLATION ===" && \
    echo "⏰ Estimated time: 1-2 minutes" && \
    start_time=$(date +%s) && \
    \
    echo "🔄 Adding Inkscape PPA repository..." && \
    apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:inkscape.dev/stable -y && \
    \
    echo "📦 Installing Inkscape..." && \
    apt-get update && \
    apt-get install -y inkscape && \
    \
    echo "🧹 Cleaning package cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    \
    end_time=$(date +%s) && \
    duration=$((end_time - start_time)) && \
    echo "✅ === INKSCAPE INSTALLATION COMPLETE === (${duration}s)"

# Install font dependencies (note: fonts-freefont-ttf already installed above)
RUN apt-get update && apt-get install -y \
    fonts-liberation \
    fontconfig && \
    fc-cache -fv && \
    rm -rf /var/lib/apt/lists/*

# Test Inkscape SVG to PDF conversion (same as your workflow)
RUN echo '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><circle cx="50" cy="50" r="40" fill="red"/></svg>' > test.svg && \
    inkscape --export-type=pdf --export-filename=test.pdf test.svg && \
    if [ -f test.pdf ]; then \
        echo "✅ Inkscape SVG to PDF conversion successful!"; \
        ls -lh test.pdf; \
    else \
        echo "❌ Inkscape SVG to PDF conversion failed."; \
        exit 1; \
    fi && \
    rm -f test.svg test.pdf

# Install Ghostscript
RUN apt-get update && apt-get install -y ghostscript && rm -rf /var/lib/apt/lists/*

# === PHASE 4: TEX LIVE INSTALLATION ===
COPY tools/dependencies/tl_packages /tmp/
RUN echo "🚀 === STARTING TEX LIVE INSTALLATION ===" && \
    echo "⏰ Estimated time: 8-12 minutes (largest phase)" && \
    echo "📊 Free disk space before: $(df -h / | tail -1 | awk '{print $4}')" && \
    start_time=$(date +%s) && \
    \
    echo "📦 Installing TeX Live prerequisites..." && \
    apt-get update && apt-get install -y \
    perl \
    wget \
    xzdec && \
    rm -rf /var/lib/apt/lists/* && \
    \
    echo "🔄 Downloading TeX Live installer (~4MB)..." && \
    wget -O /tmp/install-tl-unx.tar.gz "https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz" && \
    \
    echo "📦 Extracting TeX Live installer..." && \
    cd /tmp && tar -xzf install-tl-unx.tar.gz && \
    \
    echo "🔧 Creating TeX Live installation profile..." && \
    echo "selected_scheme scheme-infraonly" > /tmp/texlive.profile && \
    echo "tlpdbopt_install_docfiles 0" >> /tmp/texlive.profile && \
    echo "tlpdbopt_install_srcfiles 0" >> /tmp/texlive.profile && \
    echo "TEXDIR /usr/local/texlive" >> /tmp/texlive.profile && \
    echo "TEXMFCONFIG /usr/local/texlive/texmf-config" >> /tmp/texlive.profile && \
    echo "TEXMFHOME /usr/local/texlive/texmf-home" >> /tmp/texlive.profile && \
    echo "TEXMFLOCAL /usr/local/texlive/texmf-local" >> /tmp/texlive.profile && \
    echo "TEXMFSYSCONFIG /usr/local/texlive/texmf-config" >> /tmp/texlive.profile && \
    echo "TEXMFSYSVAR /usr/local/texlive/texmf-var" >> /tmp/texlive.profile && \
    echo "TEXMFVAR /usr/local/texlive/texmf-var" >> /tmp/texlive.profile && \
    \
    echo "🔄 Installing TeX Live base system..." && \
    /tmp/install-tl-*/install-tl --profile=/tmp/texlive.profile && \
    \
    echo "🔧 Setting up TeX Live PATH..." && \
    echo 'export PATH=/usr/local/texlive/bin/x86_64-linux:$PATH' >> /etc/bash.bashrc && \
    export PATH=/usr/local/texlive/bin/x86_64-linux:$PATH && \
    \
    echo "📊 Analyzing tl_packages file..." && \
    collection_count=$(grep -c '^collection-' /tmp/tl_packages) && \
    echo "📦 Found $collection_count TeX Live collections to install" && \
    \
    echo "🔄 Installing TeX Live collections..." && \
    i=1 && \
    grep -E '^collection-' /tmp/tl_packages | while read collection; do \
        echo "📦 [$i/$collection_count] Installing $collection..."; \
        tlmgr install "$collection"; \
        i=$((i+1)); \
    done && \
    \
    echo "🧹 Cleaning up TeX Live installer..." && \
    rm -rf /tmp/install-tl-* /tmp/texlive.profile /tmp/install-tl-unx.tar.gz && \
    \
    end_time=$(date +%s) && \
    duration=$((end_time - start_time)) && \
    echo "✅ === TEX LIVE INSTALLATION COMPLETE === (${duration}s)" && \
    echo "📊 Free disk space after: $(df -h / | tail -1 | awk '{print $4}')" && \
    echo "📊 TeX Live disk usage: $(du -sh /usr/local/texlive 2>/dev/null || echo 'N/A')"

# Verify TeX Live installation (simplified)
RUN export PATH=/usr/local/texlive/bin/x86_64-linux:$PATH && \
    echo "🔄 Verifying TeX Live installation..." && \
    which lualatex && which pdflatex && \
    lualatex --version | head -1 && \
    echo "✅ TeX Live verification successful"

# === PHASE 5: R INSTALLATION ===
RUN echo "🚀 === STARTING R INSTALLATION ===" && \
    echo "⏰ Estimated time: 1-2 minutes" && \
    start_time=$(date +%s) && \
    \
    echo "📦 Installing R and development packages..." && \
    apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    r-recommended && \
    \
    echo "🧹 Cleaning package cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    \
    echo "📊 R version: $(R --version | head -1)" && \
    \
    end_time=$(date +%s) && \
    duration=$((end_time - start_time)) && \
    echo "✅ === R INSTALLATION COMPLETE === (${duration}s)"

# === PHASE 6: PYTHON INSTALLATION ===
RUN echo "🚀 === STARTING PYTHON INSTALLATION ===" && \
    echo "⏰ Estimated time: 1 minute" && \
    start_time=$(date +%s) && \
    \
    echo "📦 Installing Python 3 and development packages..." && \
    apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev && \
    \
    echo "🧹 Cleaning package cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    \
    echo "📊 Python version: $(python3 --version)" && \
    echo "📊 Pip version: $(pip3 --version)" && \
    \
    end_time=$(date +%s) && \
    duration=$((end_time - start_time)) && \
    echo "✅ === PYTHON INSTALLATION COMPLETE === (${duration}s)"

# Install Quarto
RUN wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.31/quarto-1.7.31-linux-amd64.deb && \
    dpkg -i quarto-1.7.31-linux-amd64.deb && \
    rm quarto-1.7.31-linux-amd64.deb

# Create R library directory
RUN mkdir -p $R_LIBS_USER

# Copy dependency files
COPY tools/dependencies/requirements.txt /tmp/
COPY tools/dependencies/install_packages.R /tmp/
COPY docker/quarto-linux-build/verify_r_packages.R /tmp/

# Install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install -r /tmp/requirements.txt

# Install R packages (using same process as successful workflow)
RUN R --slave -e " \
    # Set options for better package installation \
    options(repos = c(CRAN = 'https://cran.rstudio.com')); \
    \
    cat('🔄 Installing R packages...\n'); \
    cat(paste('R library path:', Sys.getenv('R_LIBS_USER'), '\n')); \
    \
    # Create and set library path \
    lib_path <- Sys.getenv('R_LIBS_USER'); \
    dir.create(lib_path, showWarnings = FALSE, recursive = TRUE); \
    .libPaths(lib_path); \
    \
    # Install packages \
    cat('📦 Installing remotes package...\n'); \
    install.packages('remotes'); \
    \
    if (file.exists('/tmp/install_packages.R')) { \
      cat('📦 Installing packages from tools/dependencies/install_packages.R...\n'); \
      source('/tmp/install_packages.R'); \
    } else { \
      cat('⚠️ No tools/dependencies/install_packages.R found, installing common packages\n'); \
      pkgs <- c('rmarkdown', 'knitr', 'tidyverse', 'ggplot2', 'bookdown'); \
      cat(paste('📦 Installing packages:', paste(pkgs, collapse=', '), '\n')); \
      install.packages(pkgs); \
    }; \
    \
    cat('✅ R package installation complete\n'); \
    cat('📊 Installed packages:\n'); \
    ip <- installed.packages()[, 'Package']; \
    print(head(ip, 10)); \
    cat(paste('Total packages installed:', nrow(ip), '\n')); \
    "

# Verify R package installation (same as your workflow)
RUN echo "🔍 Verifying R package installation..." && \
    Rscript /tmp/verify_r_packages.R

# Clean up
RUN rm -rf /tmp/requirements.txt /tmp/install_packages.R /tmp/verify_r_packages.R /tmp/tl_packages

# Set working directory
WORKDIR /workspace

# Verify installations
RUN export PATH=/usr/local/texlive/bin/x86_64-linux:$PATH && \
    quarto --version && \
    python3 --version && \
    R --version && \
    lualatex --version

# Health check
RUN export PATH=/usr/local/texlive/bin/x86_64-linux:$PATH && \
    echo "✅ Container build completed successfully" && \
    echo "📊 Quarto version: $(quarto --version)" && \
    echo "📊 Python version: $(python3 --version)" && \
    echo "📊 R version: $(R --version | head -1)" && \
    echo "📊 TeX Live: $(lualatex --version | head -1)"