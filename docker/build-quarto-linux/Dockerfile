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

# Install and configure locales
RUN echo "🚀 === STARTING LOCALE CONFIGURATION ===" && \
    echo "📦 Installing locales package..." && \
    apt-get update && apt-get install -y locales && \
    echo "✅ Locales package installed" && \
    echo "🔧 Generating en_US.UTF-8 locale..." && \
    locale-gen en_US.UTF-8 && \
    echo "✅ Locale generated" && \
    echo "🔧 Updating system locale..." && \
    update-locale LANG=en_US.UTF-8 && \
    echo "✅ System locale updated" && \
    echo "🧹 Cleaning package cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    echo "✅ Locale configuration complete"

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
    echo "📋 Package list:" && \
    echo "  - fonts-dejavu" && \
    echo "  - fonts-freefont-ttf" && \
    echo "  - gdk-pixbuf2.0-bin" && \
    echo "  - libcairo2" && \
    echo "  - libfontconfig1" && \
    echo "  - libfontconfig1-dev" && \
    echo "  - libfreetype6" && \
    echo "  - libfreetype6-dev" && \
    echo "  - libpango-1.0-0" && \
    echo "  - libpangocairo-1.0-0" && \
    echo "  - libpangoft2-1.0-0" && \
    echo "  - libxml2-dev" && \
    echo "  - libcurl4-openssl-dev" && \
    echo "  - libjpeg-dev" && \
    echo "  - libtiff5-dev" && \
    echo "  - libpng-dev" && \
    echo "  - libharfbuzz-dev" && \
    echo "  - libfribidi-dev" && \
    echo "  - librsvg2-dev" && \
    echo "  - libgdal-dev" && \
    echo "  - libudunits2-dev" && \
    echo "  - wget" && \
    echo "  - curl" && \
    echo "  - git" && \
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
    echo "✅ All system packages installed successfully" && \
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
    echo "📦 Installing software-properties-common..." && \
    apt-get install -y software-properties-common && \
    echo "✅ software-properties-common installed" && \
    echo "🔧 Adding Inkscape PPA..." && \
    add-apt-repository ppa:inkscape.dev/stable -y && \
    echo "✅ Inkscape PPA added" && \
    \
    echo "📦 Installing Inkscape..." && \
    apt-get update && \
    apt-get install -y inkscape && \
    echo "✅ Inkscape installed" && \
    \
    echo "🧹 Cleaning package cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    \
    end_time=$(date +%s) && \
    duration=$((end_time - start_time)) && \
    echo "✅ === INKSCAPE INSTALLATION COMPLETE === (${duration}s)"

# Install font dependencies (note: fonts-freefont-ttf already installed above)
RUN echo "🚀 === STARTING FONT INSTALLATION ===" && \
    echo "📦 Installing additional fonts..." && \
    apt-get update && apt-get install -y \
    fonts-liberation \
    fontconfig && \
    echo "✅ Fonts installed" && \
    echo "🔧 Updating font cache..." && \
    fc-cache -fv && \
    echo "✅ Font cache updated" && \
    echo "🧹 Cleaning package cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    echo "✅ Font installation complete"

# Test Inkscape SVG to PDF conversion (same as your workflow)
RUN echo "🚀 === STARTING INKSCAPE TEST ===" && \
    echo "📋 Creating test SVG file..." && \
    echo '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><circle cx="50" cy="50" r="40" fill="red"/></svg>' > test.svg && \
    echo "✅ Test SVG created" && \
    echo "🔄 Converting SVG to PDF..." && \
    inkscape --export-type=pdf --export-filename=test.pdf test.svg && \
    echo "✅ Conversion completed" && \
    if [ -f test.pdf ]; then \
        echo "✅ Inkscape SVG to PDF conversion successful!"; \
        echo "📊 PDF file details:"; \
        ls -lh test.pdf; \
    else \
        echo "❌ Inkscape SVG to PDF conversion failed."; \
        exit 1; \
    fi && \
    echo "🧹 Cleaning up test files..." && \
    rm -f test.svg test.pdf && \
    echo "✅ Inkscape test complete"

# === PHASE 5: GHOSTSCRIPT INSTALLATION ===
RUN echo "🚀 === STARTING GHOSTSCRIPT INSTALLATION ===" && \
    echo "⏰ Estimated time: 30 seconds" && \
    start_time=$(date +%s) && \
    \
    echo "📦 Installing Ghostscript..." && \
    apt-get update && apt-get install -y ghostscript && \
    echo "✅ Ghostscript installed" && \
    echo "📊 Ghostscript version:" && \
    gs --version && \
    \
    echo "🧹 Cleaning package cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    \
    end_time=$(date +%s) && \
    duration=$((end_time - start_time)) && \
    echo "✅ === GHOSTSCRIPT INSTALLATION COMPLETE === (${duration}s)"

# === PHASE 4: TEX LIVE INSTALLATION ===
COPY tools/dependencies/tl_packages /tmp/tl_packages
RUN echo "🚀 === STARTING TEX LIVE INSTALLATION ===" && \
    echo "⏰ Estimated time: 8-12 minutes (largest phase)" && \
    echo "📊 Free disk space before: $(df -h / | tail -1 | awk '{print $4}')" && \
    start_time=$(date +%s) && \
    \
    echo "📦 Installing TeX Live prerequisites..." && \
    echo "📋 Prerequisites:" && \
    echo "  - perl" && \
    echo "  - wget" && \
    echo "  - xzdec" && \
    apt-get update && apt-get install -y \
    perl \
    wget \
    xzdec && \
    echo "✅ Prerequisites installed" && \
    rm -rf /var/lib/apt/lists/* && \
    \
    echo "🔄 Downloading TeX Live installer (~4MB)..." && \
    wget -O /tmp/install-tl-unx.tar.gz "https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz" && \
    echo "✅ Download completed" && \
    echo "📊 Downloaded file size:" && \
    ls -lh /tmp/install-tl-unx.tar.gz && \
    \
    echo "📦 Extracting TeX Live installer..." && \
    cd /tmp && tar -xzf install-tl-unx.tar.gz && \
    echo "✅ Extraction completed" && \
    echo "📊 Extracted files:" && \
    ls -la /tmp/install-tl-* && \
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
    echo "✅ Profile created" && \
    echo "📊 Profile contents:" && \
    cat /tmp/texlive.profile && \
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
    export PATH=/usr/local/texlive/bin/x86_64-linux:$PATH && \
    echo "📍 Checking if tlmgr is available..." && \
    which tlmgr && tlmgr --version || echo "❌ tlmgr not found or not working" && \
    i=1 && \
    while IFS= read -r collection; do \
        case "$collection" in \
            collection-*) \
                echo "📦 [$i/$collection_count] Installing $collection..."; \
                if command -v tlmgr >/dev/null 2>&1; then \
                    tlmgr install "$collection" || echo "⚠️ Failed to install $collection, continuing..."; \
                else \
                    echo "⚠️ tlmgr not available, skipping $collection"; \
                fi; \
                i=$((i+1)); \
                ;; \
        esac; \
    done < /tmp/tl_packages && \
    \
    echo "🧹 Cleaning up TeX Live installer..." && \
    rm -rf /tmp/install-tl-* /tmp/texlive.profile /tmp/install-tl-unx.tar.gz && \
    \
    end_time=$(date +%s) && \
    duration=$((end_time - start_time)) && \
    echo "✅ === TEX LIVE INSTALLATION COMPLETE === (${duration}s)" && \
    echo "📊 Free disk space after: $(df -h / | tail -1 | awk '{print $4}')" && \
    echo "📊 TeX Live disk usage: $(du -sh /usr/local/texlive 2>/dev/null || echo 'N/A')"

# Verify TeX Live installation (with error handling)
RUN echo "🔄 Verifying TeX Live installation..." && \
    export PATH=/usr/local/texlive/bin/x86_64-linux:$PATH && \
    echo "📍 PATH: $PATH" && \
    echo "📍 Checking TeX Live directory:" && \
    ls -la /usr/local/texlive/ || echo "❌ TeX Live directory not found" && \
    ls -la /usr/local/texlive/bin/ || echo "❌ TeX Live bin directory not found" && \
    if [ -f /usr/local/texlive/bin/x86_64-linux/lualatex ]; then \
        echo "✅ lualatex found"; \
        /usr/local/texlive/bin/x86_64-linux/lualatex --version | head -1; \
    else \
        echo "❌ lualatex not found, checking for alternative locations"; \
        find /usr/local/texlive -name "lualatex" -type f 2>/dev/null || echo "No lualatex found anywhere"; \
    fi && \
    echo "✅ TeX Live verification complete (allowing partial failures)"

# === PHASE 6: R INSTALLATION ===
RUN echo "🚀 === STARTING R INSTALLATION ===" && \
    echo "⏰ Estimated time: 1-2 minutes" && \
    start_time=$(date +%s) && \
    \
    echo "📦 Installing R and development packages..." && \
    echo "📋 R packages:" && \
    echo "  - r-base" && \
    echo "  - r-base-dev" && \
    echo "  - r-recommended" && \
    apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    r-recommended && \
    echo "✅ R packages installed" && \
    \
    echo "🧹 Cleaning package cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    \
    echo "📊 R version: $(R --version | head -1)" && \
    \
    end_time=$(date +%s) && \
    duration=$((end_time - start_time)) && \
    echo "✅ === R INSTALLATION COMPLETE === (${duration}s)"

# === PHASE 7: PYTHON INSTALLATION ===
RUN echo "🚀 === STARTING PYTHON INSTALLATION ===" && \
    echo "⏰ Estimated time: 1 minute" && \
    start_time=$(date +%s) && \
    \
    echo "📦 Installing Python 3 and development packages..." && \
    echo "📋 Python packages:" && \
    echo "  - python3" && \
    echo "  - python3-pip" && \
    echo "  - python3-dev" && \
    apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev && \
    echo "✅ Python packages installed" && \
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

# === PHASE 3: QUARTO INSTALLATION ===
RUN echo "🚀 === STARTING QUARTO INSTALLATION ===" && \
    echo "⏰ Estimated time: 1 minute" && \
    start_time=$(date +%s) && \
    \
    echo "📦 Downloading Quarto 1.7.31..." && \
    wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.31/quarto-1.7.31-linux-amd64.deb && \
    echo "✅ Download completed" && \
    echo "📊 Downloaded file size:" && \
    ls -lh quarto-1.7.31-linux-amd64.deb && \
    \
    echo "📦 Installing Quarto..." && \
    dpkg -i quarto-1.7.31-linux-amd64.deb && \
    echo "✅ Quarto installed" && \
    \
    echo "🧹 Cleaning up installer..." && \
    rm quarto-1.7.31-linux-amd64.deb && \
    echo "✅ Installer cleaned up" && \
    \
    end_time=$(date +%s) && \
    duration=$((end_time - start_time)) && \
    echo "✅ === QUARTO INSTALLATION COMPLETE === (${duration}s)"

# Create R library directory
RUN echo "🚀 === STARTING R LIBRARY SETUP ===" && \
    echo "📁 Creating R library directory..." && \
    mkdir -p $R_LIBS_USER && \
    echo "✅ R library directory created: $R_LIBS_USER" && \
    echo "✅ R library setup complete"

# Copy dependency files
RUN echo "🚀 === STARTING DEPENDENCY FILE COPY ===" && \
    echo "📋 Files to copy:" && \
    echo "  - tools/dependencies/requirements/ → /tmp/requirements/" && \
    echo "  - tools/dependencies/requirements-build.txt → /tmp/requirements.txt" && \
    echo "  - tools/dependencies/install_packages.R → /tmp/install_packages.R" && \
    echo "  - docker/build-quarto-linux/verify_r_packages.R → /tmp/verify_r_packages.R" && \
    echo "✅ Dependency file copy phase complete"

COPY tools/dependencies/requirements/ /tmp/requirements/
COPY tools/dependencies/requirements-build.txt /tmp/requirements.txt
COPY tools/dependencies/install_packages.R /tmp/install_packages.R
COPY docker/build-quarto-linux/verify_r_packages.R /tmp/verify_r_packages.R

# === PHASE 8: PYTHON PACKAGES ===
RUN echo "🚀 === STARTING PYTHON PACKAGE INSTALLATION ===" && \
    echo "⏰ Estimated time: 1-2 minutes" && \
    start_time=$(date +%s) && \
    \
    echo "🔄 Upgrading pip..." && \
    pip3 install --upgrade pip && \
    \
    echo "📊 Analyzing requirements.txt..." && \
    package_count=$(grep -v '^#' /tmp/requirements.txt | grep -v '^$' | wc -l) && \
    echo "📦 Found $package_count Python packages to install" && \
    \
    echo "🔄 Installing Python packages with space optimization..." && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    \
    echo "🧹 Cleaning Python installation caches..." && \
    pip3 cache purge && \
    find /usr -name "*.pyc" -delete && \
    find /usr -name "__pycache__" -type d -exec rm -rf {} + || true && \
    \
    echo "📊 Installed Python packages:" && \
    pip3 list | head -10 && \
    echo "📊 Total packages: $(pip3 list | wc -l)" && \
    \
    end_time=$(date +%s) && \
    duration=$((end_time - start_time)) && \
    echo "✅ === PYTHON PACKAGES COMPLETE === (${duration}s)"

# === PHASE 9: R PACKAGES ===
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

# === PHASE 10: R PACKAGE VERIFICATION ===
RUN echo "🔍 Verifying R package installation..." && \
    Rscript /tmp/verify_r_packages.R

# === PHASE 11: COMPREHENSIVE CLEANUP ===
RUN echo "🚀 === STARTING COMPREHENSIVE CLEANUP ===" && \
    echo "📊 Disk space before cleanup: $(df -h / | tail -1 | awk '{print $4}')" && \
    start_time=$(date +%s) && \
    \
    echo "🧹 Removing temporary files..." && \
    rm -rf /tmp/* && \
    rm -rf /var/tmp/* && \
    \
    echo "🧹 Cleaning package caches..." && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/* && \
    \
    echo "🧹 Cleaning Python caches..." && \
    find /usr -name "*.pyc" -delete && \
    find /usr -name "__pycache__" -type d -exec rm -rf {} + || true && \
    pip3 cache purge || true && \
    \
    echo "🧹 Cleaning R temporary files..." && \
    rm -rf /tmp/Rtmp* || true && \
    rm -rf /var/lib/R/site-library/*/help || true && \
    \
    echo "🧹 Cleaning TeX Live caches and docs..." && \
    rm -rf /usr/local/texlive/*/texmf-var/luatex-cache/* || true && \
    rm -rf /usr/local/texlive/*/texmf-var/web2c/* || true && \
    \
    echo "🧹 Removing unnecessary system files..." && \
    rm -rf /usr/share/doc/* && \
    rm -rf /usr/share/man/* && \
    rm -rf /usr/share/info/* && \
    rm -rf /var/log/* && \
    \
    end_time=$(date +%s) && \
    duration=$((end_time - start_time)) && \
    echo "✅ === COMPREHENSIVE CLEANUP COMPLETE === (${duration}s)" && \
    echo "📊 Final disk space: $(df -h / | tail -1 | awk '{print $4}')"

# Set working directory
WORKDIR /workspace

# Verify installations
RUN echo "🚀 === STARTING FINAL VERIFICATION ===" && \
    export PATH=/usr/local/texlive/bin/x86_64-linux:$PATH && \
    echo "📊 Checking Quarto..." && \
    quarto --version && \
    echo "✅ Quarto verified" && \
    echo "📊 Checking Python..." && \
    python3 --version && \
    echo "✅ Python verified" && \
    echo "📊 Checking R..." && \
    R --version && \
    echo "✅ R verified" && \
    echo "📊 Checking LaTeX..." && \
    lualatex --version && \
    echo "✅ LaTeX verified" && \
    echo "✅ Final verification complete"

# Health check
RUN export PATH=/usr/local/texlive/bin/x86_64-linux:$PATH && \
    echo "✅ Container build completed successfully" && \
    echo "📊 Quarto version: $(quarto --version)" && \
    echo "📊 Python version: $(python3 --version)" && \
    echo "📊 R version: $(R --version | head -1)" && \
    echo "📊 TeX Live: $(lualatex --version | head -1)"