# MLSysBook Quarto Build Container
# Based on Ubuntu 22.04 with all dependencies pre-installed
# This container eliminates the 30-45 minute setup time for Linux builds

FROM ubuntu:22.04

# === PATH CONFIGURATION ===
# Paths are hardcoded since they're stable and ARG scope causes issues in multi-layer builds

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV R_LIBS_USER=/usr/local/lib/R/library
ENV QUARTO_LOG_LEVEL=INFO
ENV PYTHONIOENCODING=utf-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV PATH=/usr/local/texlive/bin/x86_64-linux:$PATH

# === PHASE 0: COPY DEPENDENCY FILES EARLY (for better cache efficiency) ===
# Copy dependency files early for better cache efficiency
COPY book/tools/dependencies/requirements.txt /tmp/requirements.txt
COPY book/tools/dependencies/install_packages.R /tmp/install_packages.R
COPY book/tools/dependencies/tl_packages /tmp/tl_packages
COPY book/docker/linux/verify_r_packages.R /tmp/verify_r_packages.R
COPY book/docker/linux/install-texlive-base.sh /opt/install-texlive-base.sh

# Note: Python packages are installed later in Phase 10

# Early cleanup
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# === PHASE 1: LOCALE CONFIGURATION ===
RUN echo "🚀 === STARTING LOCALE CONFIGURATION ===" && \
    echo "🔍 Checking system readiness..." && \
    if [ -f /etc/os-release ]; then \
        echo "✅ OS release file found"; \
        cat /etc/os-release | grep PRETTY_NAME; \
    else \
        echo "❌ OS release file not found"; \
        exit 1; \
    fi && \
    echo "📦 Installing locales package..." && \
    apt-get update && apt-get install -y locales && \
    echo "📦 Locales package installed" && \
    echo "🔧 Generating en_US.UTF-8 locale..." && \
    locale-gen en_US.UTF-8 && \
    echo "📄 Locale generated" && \
    echo "🔧 Updating system locale..." && \
    update-locale LANG=en_US.UTF-8 && \
    echo "🔧 System locale updated" && \
    echo "🧹 Cleaning package cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    echo "✅ Locale configuration complete"

# === PHASE 2: SYSTEM DEPENDENCIES ===
RUN echo "🚀 === STARTING SYSTEM DEPENDENCIES INSTALLATION ===" && \
    echo "⏰ Estimated time: 2-3 minutes" && \
    echo "📊 Free disk space: $$(df -h / | tail -1 | awk '{print $$4}')" && \
    start_time=$$(date +%s) && \
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
    echo "  - librsvg2-bin" && \
    echo "  - libgdal-dev" && \
    echo "  - libudunits2-dev" && \
    echo "  - wget" && \
    echo "  - curl" && \
    echo "  - git" && \
    echo "  - cmake" && \
    echo "  - pkg-config" && \
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
    librsvg2-bin \
    libgdal-dev \
    libudunits2-dev \
    wget \
    curl \
    git \
    cmake \
    pkg-config && \
    echo "📦 All system packages installed successfully" && \
    \
    echo "🔍 Verifying critical packages..." && \
    if command -v wget >/dev/null 2>&1; then \
        echo "📦 wget available"; \
    else \
        echo "❌ wget not found"; \
        exit 1; \
    fi && \
    if command -v curl >/dev/null 2>&1; then \
        echo "📦 curl available"; \
    else \
        echo "❌ curl not found"; \
        exit 1; \
    fi && \
    if command -v git >/dev/null 2>&1; then \
        echo "📦 git available"; \
    else \
        echo "❌ git not found"; \
        exit 1; \
    fi && \
    if command -v rsvg-convert >/dev/null 2>&1; then \
        echo "📦 rsvg-convert available"; \
    else \
        echo "❌ rsvg-convert not found"; \
        exit 1; \
    fi && \
    \
    echo "🧹 Cleaning package cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    \
    end_time=$$(date +%s) && \
    duration=$$((end_time - start_time)) && \
    echo "✅ === SYSTEM DEPENDENCIES COMPLETE === ($${duration}s)" && \
    echo "📊 Free disk space: $$(df -h / | tail -1 | awk '{print $$4}')"

# === PHASE 3: INKSCAPE INSTALLATION ===
RUN echo "🚀 === STARTING INKSCAPE INSTALLATION ===" && \
    echo "⏰ Estimated time: 1-2 minutes" && \
    start_time=$$(date +%s) && \
    \
    echo "🔄 Adding Inkscape PPA repository..." && \
    apt-get update && \
    echo "📦 Installing software-properties-common..." && \
    apt-get install -y software-properties-common && \
    echo "📦 software-properties-common installed" && \
    echo "🔧 Adding Inkscape PPA..." && \
    add-apt-repository ppa:inkscape.dev/stable -y && \
    echo "📦 Inkscape PPA added" && \
    \
    echo "📦 Installing Inkscape..." && \
    apt-get update && \
    apt-get install -y inkscape && \
    echo "📦 Inkscape installed" && \
    \
    echo "🧹 Cleaning package cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    \
    end_time=$$(date +%s) && \
    duration=$$((end_time - start_time)) && \
    echo "✅ === INKSCAPE INSTALLATION COMPLETE === ($${duration}s)"

# Install font dependencies (note: fonts-freefont-ttf already installed above)
RUN echo "🚀 === STARTING FONT INSTALLATION ===" && \
    echo "📦 Installing additional fonts..." && \
    apt-get update && apt-get install -y \
    fonts-liberation \
    fontconfig && \
    echo "📦 Fonts installed" && \
    echo "🔧 Updating font cache..." && \
    fc-cache -fv && \
    echo "📄 Font cache updated" && \
    echo "🧹 Cleaning package cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    echo "✅ Font installation complete"

# Test Inkscape SVG to PDF conversion (same as your workflow)
RUN echo "🚀 === STARTING INKSCAPE TEST ===" && \
    echo "📋 Creating test SVG file..." && \
    echo '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><circle cx="50" cy="50" r="40" fill="red"/></svg>' > test.svg && \
    echo "📄 Test SVG created" && \
    echo "🔄 Converting SVG to PDF..." && \
    inkscape --export-type=pdf --export-filename=test.pdf test.svg && \
    echo "📦 Conversion completed" && \
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

# === PHASE 4: GHOSTSCRIPT INSTALLATION ===
RUN echo "🚀 === STARTING GHOSTSCRIPT INSTALLATION ===" && \
    echo "⏰ Estimated time: 30 seconds" && \
    start_time=$$(date +%s) && \
    \
    echo "📦 Installing Ghostscript..." && \
    apt-get update && apt-get install -y ghostscript && \
    echo "📦 Ghostscript installed" && \
    echo "📊 Ghostscript version:" && \
    gs --version && \
    \
    echo "🧹 Cleaning package cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    \
    end_time=$$(date +%s) && \
    duration=$$((end_time - start_time)) && \
    echo "✅ === GHOSTSCRIPT INSTALLATION COMPLETE === ($${duration}s)"

# === PHASE 5: TEX LIVE INSTALLATION ===
# Step 5.1: Install TeX Live prerequisites
RUN echo "🚀 === STARTING TEX LIVE INSTALLATION ===" && \
    echo "⏰ Estimated time: 8-12 minutes (largest phase)" && \
    echo "📊 Free disk space before: $$(df -h / | tail -1 | awk '{print $$4}')" && \
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
    echo "📦 Prerequisites installed" && \
    \
    echo "🔍 Verifying prerequisites..." && \
    if command -v perl >/dev/null 2>&1; then \
        echo "📦 perl available"; \
    else \
        echo "❌ perl not found"; \
        exit 1; \
    fi && \
    if command -v wget >/dev/null 2>&1; then \
        echo "📦 wget available"; \
    else \
        echo "❌ wget not found"; \
        exit 1; \
    fi && \
    if command -v xzdec >/dev/null 2>&1; then \
        echo "📦 xzdec available"; \
    else \
        echo "❌ xzdec not found"; \
        exit 1; \
    fi && \
    \
    rm -rf /var/lib/apt/lists/* && \
    echo "✅ TeX Live prerequisites installed"

# Step 5.2+5.3: install-tl download + base install in ONE layer.
# GHA buildx cache can otherwise reuse an old "download only" layer (TL year N) while
# a later layer passes -repository for year N+1 — install-tl fails in seconds.
# Bump TEXLIVE_INSTALLER_REVISION when the annual CTAN rollover requires a fresh
# install-tl-unx.tar.gz for this image.
# install-tl -repository must point at a base URL whose tlpdb is at .../tlpkg/texlive.tlpdb
# (curl -I on that file should be 200). As of 2026-04, mirror.ctan install-tl-unx still unpacks
# to install-tl-2025*; main tlnet is 2026, so 2025 historic tlnet-final is required for a match.
# TeX base install is a separate POSIX script (avoids long RUN + buildx $/() handling on /bin/sh).
ARG TEXLIVE_INSTALLER_REVISION=6
RUN chmod 0755 /opt/install-texlive-base.sh && /opt/install-texlive-base.sh ${TEXLIVE_INSTALLER_REVISION}

# Step 5.4: Configure TeX Live PATH
RUN echo "🔧 Setting up TeX Live PATH..." && \
    echo 'export PATH=/usr/local/texlive/bin/x86_64-linux:$$PATH' >> /etc/bash.bashrc && \
    export PATH=/usr/local/texlive/bin/x86_64-linux:$$PATH && \
    echo "🔧 TeX Live PATH configured" && \
    \
    echo "📍 Checking if tlmgr is available..." && \
    if command -v tlmgr >/dev/null 2>&1; then \
        echo "📦 tlmgr available"; \
        tlmgr --version | head -1; \
    else \
        echo "❌ tlmgr not found or not working"; \
        exit 1; \
    fi && \
    echo "✅ TeX Live PATH and tlmgr verified"

# Step 5.5: Install TeX Live packages with improved error handling
RUN echo "📊 Analyzing tl_packages file..." && \
    collection_count=$$(grep -c '^collection-' /tmp/tl_packages) && \
    echo "📦 Found $collection_count TeX Live collections to install" && \
    \
    echo "🔄 Installing TeX Live collections with retry logic..." && \
    export PATH=/usr/local/texlive/bin/x86_64-linux:$$PATH && \
    TLMGR_REPO=$$(tr -d '\n' < /opt/mlsysbook-texlive-tlnet-url) && \
    tlmgr option repository "$$TLMGR_REPO" && \
    i=1 && \
    failed_packages="" && \
    while IFS= read -r collection; do \
        case "$collection" in \
            collection-*) \
                echo "📦 [$i/$collection_count] Installing $collection..."; \
                if command -v tlmgr >/dev/null 2>&1; then \
                    success=false; \
                    for retry in 1 2; do \
                        if tlmgr install "$collection"; then \
                            echo "✅ Successfully installed $collection"; \
                            success=true; \
                            break; \
                        else \
                            echo "❌ Attempt $retry failed for $collection"; \
                            if [ $retry -lt 2 ]; then \
                                echo "⏳ Retrying in 5 seconds..."; \
                                sleep 5; \
                            fi; \
                        fi; \
                    done; \
                    if [ "$success" = false ]; then \
                        echo "⚠️ Failed to install $collection after retries, continuing..."; \
                        failed_packages="$failed_packages $collection"; \
                    fi; \
                else \
                    echo "⚠️ tlmgr not available, skipping $collection"; \
                fi; \
                i=$$((i+1)); \
                ;; \
        esac; \
    done < /tmp/tl_packages && \
    if [ -n "$failed_packages" ]; then \
        echo "⚠️ Some packages failed to install:$failed_packages"; \
        echo "📋 This may not be critical for basic functionality"; \
    fi && \
    echo "✅ TeX Live packages installation completed"

# Step 5.6: TeX Live cleanup and completion
RUN echo "🧹 Cleaning up TeX Live installer..." && \
    rm -rf /tmp/install-tl-* /tmp/texlive.profile /tmp/install-tl-unx.tar.gz && \
    echo "✅ === TEX LIVE INSTALLATION COMPLETE ===" && \
    echo "📊 Free disk space after: $$(df -h / | tail -1 | awk '{print $$4}')" && \
    echo "📊 TeX Live disk usage: $$(du -sh /usr/local/texlive 2>/dev/null || echo 'N/A')"

# Verify TeX Live installation (with error handling)
RUN echo "🔄 Verifying TeX Live installation..." && \
    export PATH=/usr/local/texlive/bin/x86_64-linux:$$PATH && \
    echo "📍 PATH: $$PATH" && \
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

# === PHASE 6: R INSTALLATION (R 4.5 from CRAN) ===
RUN echo "🚀 === STARTING R INSTALLATION ===" && \
    echo "⏰ Estimated time: 1-2 minutes" && \
    start_time=$$(date +%s) && \
    \
    echo "📦 Adding CRAN repository for R 4.5..." && \
    apt-get update && apt-get install -y --no-install-recommends software-properties-common dirmngr && \
    wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc > /dev/null && \
    add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" && \
    apt-get update && \
    \
    echo "📦 Installing R and development packages..." && \
    echo "📋 R packages:" && \
    echo "  - r-base" && \
    echo "  - r-base-dev" && \
    echo "  - r-recommended" && \
    apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    r-recommended && \
    if [ $$? -ne 0 ]; then \
        echo "❌ Failed to install R packages"; \
        exit 1; \
    fi && \
    echo "📦 R packages installed" && \
    \
    echo "🧹 Cleaning package cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    \
    echo "📊 R version: $$(R --version | head -1)" && \
    if [ $$? -ne 0 ]; then \
        echo "❌ Failed to verify R installation"; \
        exit 1; \
    fi && \
    \
    end_time=$$(date +%s) && \
    duration=$$((end_time - start_time)) && \
    echo "✅ === R INSTALLATION COMPLETE === ($${duration}s)"

# R may omit R_LIBS_USER from .libPaths() if the directory is missing, so install and verify
# can disagree on where packages were written. Create it before any Rscript package step.
RUN mkdir -p $R_LIBS_USER

# Fail fast: building CRAN source packages (e.g. fs → libuv) needs cmake. Without it,
# the R step runs 30–60+ minutes then dies at verify. See e12d7dc4a8.
RUN echo "🔎 Pre-flight: R build toolchain (cmake, pkg-config)…" && \
    test -d "$R_LIBS_USER" || { echo "❌ R_LIBS_USER dir missing: $R_LIBS_USER"; exit 1; } && \
    command -v cmake >/dev/null 2>&1 || { echo "❌ cmake not in PATH (required for R package fs, needed by rmarkdown/tidyverse)"; exit 1; } && \
    command -v pkg-config >/dev/null 2>&1 || { echo "❌ pkg-config not in PATH"; exit 1; } && \
    echo "✅ Pre-flight OK"

# === PHASE 7: R PACKAGE INSTALLATION ===
RUN echo "🚀 === STARTING R PACKAGE INSTALLATION ===" && \
    echo "⏰ Cold build: this step compiles many CRAN packages; expect 20–60+ min (cache shortens reruns)" && \
    start_time=$$(date +%s) && \
    \
    echo "Running R setup script with the following files:" && \
    echo "  - /tmp/install_packages.R (from book/tools/dependencies/install_packages.R)" && \
    echo "  - /tmp/verify_r_packages.R (from book/docker/linux/verify_r_packages.R)" && \
    \
    echo "📦 Installing R packages..." && \
    Rscript /tmp/install_packages.R && \
    if [ $$? -ne 0 ]; then \
        echo "❌ Failed to install R packages"; \
        exit 1; \
    fi && \
    echo "✅ R packages installed successfully" && \
    \
    echo "🔍 Verifying R packages..." && \
    Rscript /tmp/verify_r_packages.R && \
    if [ $$? -ne 0 ]; then \
        echo "❌ Failed to verify R packages"; \
        exit 1; \
    fi && \
    echo "✅ R packages verified successfully" && \
    \
    echo "🧹 Cleaning up R installation files..." && \
    rm -f /tmp/install_packages.R /tmp/verify_r_packages.R && \
    \
    end_time=$$(date +%s) && \
    duration=$$((end_time - start_time)) && \
    echo "✅ === R PACKAGE INSTALLATION COMPLETE === ($${duration}s)"

# === PHASE 8: PYTHON INSTALLATION ===
RUN echo "🚀 === STARTING PYTHON INSTALLATION ===" && \
    echo "⏰ Estimated time: 1 minute" && \
    start_time=$$(date +%s) && \
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
    if [ $$? -ne 0 ]; then \
        echo "❌ Failed to install Python packages"; \
        exit 1; \
    fi && \
    echo "📦 Python packages installed" && \
    \
    echo "🧹 Cleaning package cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    \
    echo "📊 Python version: $$(python3 --version)" && \
    if [ $$? -ne 0 ]; then \
        echo "❌ Failed to verify Python installation"; \
        exit 1; \
    fi && \
    echo "📊 Pip version: $$(pip3 --version)" && \
    if [ $$? -ne 0 ]; then \
        echo "❌ Failed to verify pip installation"; \
        exit 1; \
    fi && \
    \
    end_time=$$(date +%s) && \
    duration=$$((end_time - start_time)) && \
    echo "✅ === PYTHON INSTALLATION COMPLETE === ($${duration}s)"

# === PHASE 9: QUARTO INSTALLATION ===
RUN echo "🚀 === STARTING QUARTO INSTALLATION ===" && \
    echo "⏰ Estimated time: 1 minute" && \
    start_time=$$(date +%s) && \
    \
    echo "📦 Downloading Quarto 1.9.27..." && \
    wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v1.9.27/quarto-1.9.27-linux-amd64.deb && \
    if [ $$? -ne 0 ]; then \
        echo "❌ Failed to download Quarto"; \
        exit 1; \
    fi && \
    echo "📥 Download completed" && \
    echo "📊 Downloaded file size:" && \
    ls -lh quarto-1.9.27-linux-amd64.deb && \
    \
    echo "📦 Installing Quarto..." && \
    dpkg -i quarto-1.9.27-linux-amd64.deb && \
    if [ $$? -ne 0 ]; then \
        echo "❌ Failed to install Quarto"; \
        exit 1; \
    fi && \
    echo "📦 Quarto installed" && \
    \
    echo "🧹 Cleaning up installer..." && \
    rm quarto-1.9.27-linux-amd64.deb && \
    echo "🧹 Installer cleaned up" && \
    \
    end_time=$$(date +%s) && \
    duration=$$((end_time - start_time)) && \
    echo "✅ === QUARTO INSTALLATION COMPLETE === ($${duration}s)"

# Create R library directory
RUN echo "🚀 === STARTING R LIBRARY SETUP ===" && \
    echo "📁 Creating R library directory..." && \
    mkdir -p $R_LIBS_USER && \
    echo "✅ R library directory created: $R_LIBS_USER" && \
    echo "✅ R library setup complete"



# === PHASE 10: PYTHON PACKAGES ===
RUN echo "🚀 === STARTING PYTHON PACKAGE INSTALLATION ===" && \
    echo "⏰ Estimated time: 1-2 minutes" && \
    start_time=$$(date +%s) && \
    \
    echo "🔄 Upgrading pip..." && \
    pip3 install --upgrade pip && \
    \
    echo "📊 Analyzing requirements.txt..." && \
    package_count=$$(grep -v '^#' /tmp/requirements.txt | grep -v '^$' | wc -l) && \
    echo "📦 Found $package_count Python packages to install" && \
    \
    echo "🔄 Installing Python packages with space optimization..." && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    if [ $$? -ne 0 ]; then \
        echo "❌ Failed to install Python packages"; \
        exit 1; \
    fi && \
    \
    echo "🧹 Cleaning Python installation caches..." && \
    pip3 cache purge && \
    find /usr -name "*.pyc" -delete && \
    find /usr -name "__pycache__" -type d -exec rm -rf {} + || true && \
    \
    echo "📊 Installed Python packages:" && \
    pip3 list | head -10 && \
    echo "📊 Total packages: $$(pip3 list | wc -l)" && \
    \
    end_time=$$(date +%s) && \
    duration=$$((end_time - start_time)) && \
    echo "✅ === PYTHON PACKAGES COMPLETE === ($${duration}s)"

# === PHASE 9: R PACKAGES ===
# Step 9.1: Set up R environment and install remotes
RUN echo "🚀 === STEP 9.1: SETTING UP R ENVIRONMENT ===" && \
    R --slave -e " \
    options(repos = c(CRAN = 'https://cran.rstudio.com')); \
    cat('🔄 Setting up R environment...\n'); \
    cat(paste('R library path:', Sys.getenv('R_LIBS_USER'), '\n')); \
    lib_path <- Sys.getenv('R_LIBS_USER'); \
    dir.create(lib_path, showWarnings = FALSE, recursive = TRUE); \
    .libPaths(lib_path); \
    cat('📦 Installing remotes package...\n'); \
    install.packages('remotes'); \
    cat('✅ R environment setup complete\n'); \
    " && \
    if [ $$? -ne 0 ]; then \
        echo "❌ Failed to set up R environment"; \
        exit 1; \
    fi && \
    echo "✅ R environment setup successful"

# Step 9.2: Install R packages from install_packages.R or fallback
RUN echo "🚀 === STEP 9.2: INSTALLING R PACKAGES ===" && \
    R --slave -e " \
    options(repos = c(CRAN = 'https://cran.rstudio.com')); \
    lib_path <- Sys.getenv('R_LIBS_USER'); \
    .libPaths(lib_path); \
    if (file.exists('/tmp/install_packages.R')) { \
      cat('📦 Installing packages from tools/dependencies/install_packages.R...\n'); \
      source('/tmp/install_packages.R'); \
    } else { \
      cat('⚠️ No tools/dependencies/install_packages.R found, installing common packages\n'); \
      pkgs <- c('rmarkdown', 'knitr', 'tidyverse', 'ggplot2', 'bookdown'); \
      cat(paste('📦 Installing packages:', paste(pkgs, collapse=', '), '\n')); \
      install.packages(pkgs); \
    }; \
    cat('✅ R package installation complete\n'); \
    " && \
    if [ $$? -ne 0 ]; then \
        echo "❌ Failed to install R packages"; \
        exit 1; \
    fi && \
    echo "✅ R packages installed successfully"

# Step 9.3: Verify R package installation
RUN echo "🚀 === STEP 9.3: VERIFYING R PACKAGES ===" && \
    R --slave -e " \
    lib_path <- Sys.getenv('R_LIBS_USER'); \
    .libPaths(lib_path); \
    cat('📊 Installed packages:\n'); \
    ip <- installed.packages()[, 'Package']; \
    print(head(ip, 10)); \
    cat(paste('Total packages installed:', nrow(ip), '\n')); \
    " && \
    if [ $$? -ne 0 ]; then \
        echo "❌ Failed to verify R packages"; \
        exit 1; \
    fi && \
    echo "✅ R package verification successful"

# === PHASE 11: R PACKAGE VERIFICATION ===
# Note: R package verification is now done immediately after R package installation (Phase 6.5)

# === PHASE 12: COMPREHENSIVE CLEANUP ===
RUN echo "🚀 === STARTING COMPREHENSIVE CLEANUP ===" && \
    echo "📊 Disk space before cleanup: $$(df -h / | tail -1 | awk '{print $$4}')" && \
    start_time=$$(date +%s) && \
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
    end_time=$$(date +%s) && \
    duration=$$((end_time - start_time)) && \
    echo "✅ === COMPREHENSIVE CLEANUP COMPLETE === ($${duration}s)" && \
    echo "📊 Final disk space: $$(df -h / | tail -1 | awk '{print $$4}')"

# Set working directory
WORKDIR /workspace

# === PHASE 13: FINAL VERIFICATION ===
RUN echo "🚀 === STARTING FINAL VERIFICATION ===" && \
    export PATH=/usr/local/texlive/bin/x86_64-linux:$$PATH && \
    echo "📊 Checking Quarto..." && \
    if command -v quarto >/dev/null 2>&1; then \
        quarto --version && \
        echo "✅ Quarto verified"; \
    else \
        echo "❌ Quarto not found in PATH"; \
        echo "📍 Checking for quarto in common locations:"; \
        find /usr -name "quarto" -type f 2>/dev/null || echo "No quarto found"; \
        exit 1; \
    fi && \
    echo "📊 Checking Python..." && \
    if command -v python3 >/dev/null 2>&1; then \
        python3 --version && \
        echo "✅ Python verified"; \
    else \
        echo "❌ Python3 not found"; \
        exit 1; \
    fi && \
    echo "📊 Checking R..." && \
    if command -v R >/dev/null 2>&1; then \
        R --version && \
        echo "✅ R verified"; \
    else \
        echo "❌ R not found"; \
        exit 1; \
    fi && \
    echo "📊 Checking LaTeX..." && \
    if command -v lualatex >/dev/null 2>&1; then \
        lualatex --version && \
        echo "✅ LaTeX verified"; \
    else \
        echo "❌ lualatex not found"; \
        echo "📍 Checking for lualatex in TeX Live:"; \
        find /usr/local/texlive -name "lualatex" -type f 2>/dev/null || echo "No lualatex found"; \
        exit 1; \
    fi && \
    echo "✅ Final verification complete"

# Health check
RUN export PATH=/usr/local/texlive/bin/x86_64-linux:$$PATH && \
    echo "✅ Container build completed successfully" && \
    echo "📊 Quarto version: $$(quarto --version)" && \
    echo "📊 Python version: $$(python3 --version)" && \
    echo "📊 R version: $$(R --version | head -1)" && \
    echo "📊 TeX Live: $$(lualatex --version | head -1)"
