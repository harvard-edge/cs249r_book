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

# === PHASE 2: INKSCAPE INSTALLATION === (COMMENTED OUT FOR TESTING)
# RUN echo "🚀 === STARTING INKSCAPE INSTALLATION ===" && \
#     echo "⏰ Estimated time: 1-2 minutes" && \
#     start_time=$(date +%s) && \
#     \
#     echo "🔄 Adding Inkscape PPA repository..." && \
#     apt-get update && \
#     apt-get install -y software-properties-common && \
#     add-apt-repository ppa:inkscape.dev/stable -y && \
#     \
#     echo "📦 Installing Inkscape..." && \
#     apt-get update && \
#     apt-get install -y inkscape && \
#     \
#     echo "🧹 Cleaning package cache..." && \
#     rm -rf /var/lib/apt/lists/* && \
#     \
#     end_time=$(date +%s) && \
#     duration=$((end_time - start_time)) && \
#     echo "✅ === INKSCAPE INSTALLATION COMPLETE === (${duration}s)"

# Install font dependencies (note: fonts-freefont-ttf already installed above) (COMMENTED OUT FOR TESTING)
# RUN apt-get update && apt-get install -y \
#     fonts-liberation \
#     fontconfig && \
#     fc-cache -fv && \
#     rm -rf /var/lib/apt/lists/*

# Test Inkscape SVG to PDF conversion (same as your workflow) (COMMENTED OUT FOR TESTING)
# RUN echo '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><circle cx="50" cy="50" r="40" fill="red"/></svg>' > test.svg && \
#     inkscape --export-type=pdf --export-filename=test.pdf test.svg && \
#     if [ -f test.pdf ]; then \
#         echo "✅ Inkscape SVG to PDF conversion successful!"; \
#         ls -lh test.pdf; \
#     else \
#         echo "❌ Inkscape SVG to PDF conversion failed."; \
#         exit 1; \
#     fi && \
#     rm -f test.svg test.pdf

# === PHASE 5: GHOSTSCRIPT INSTALLATION === (COMMENTED OUT FOR TESTING)
# RUN echo "🚀 === STARTING GHOSTSCRIPT INSTALLATION ===" && \
#     echo "⏰ Estimated time: 30 seconds" && \
#     start_time=$(date +%s) && \
#     \
#     echo "📦 Installing Ghostscript..." && \
#     apt-get update && apt-get install -y ghostscript && \
#     \
#     echo "🧹 Cleaning package cache..." && \
#     rm -rf /var/lib/apt/lists/* && \
#     \
#     end_time=$(date +%s) && \
#     duration=$((end_time - start_time)) && \
#     echo "✅ === GHOSTSCRIPT INSTALLATION COMPLETE === (${duration}s)"

# === PHASE 4: TEX LIVE INSTALLATION === (COMMENTED OUT FOR TESTING - BIGGEST TIME CONSUMER)
# COPY tools/dependencies/tl_packages /tmp/tl_packages
# RUN echo "🚀 === STARTING TEX LIVE INSTALLATION ===" && \
#     echo "⏰ Estimated time: 8-12 minutes (largest phase)" && \
#     echo "📊 Free disk space before: $(df -h / | tail -1 | awk '{print $4}')" && \
#     start_time=$(date +%s) && \
#     \
#     echo "📦 Installing TeX Live prerequisites..." && \
#     apt-get update && apt-get install -y \
#     perl \
#     wget \
#     xzdec && \
#     rm -rf /var/lib/apt/lists/* && \
#     \
#     echo "🔄 Downloading TeX Live installer (~4MB)..." && \
#     wget -O /tmp/install-tl-unx.tar.gz "https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz" && \
#     \
#     echo "📦 Extracting TeX Live installer..." && \
#     cd /tmp && tar -xzf install-tl-unx.tar.gz && \
#     \
#     echo "🔧 Creating TeX Live installation profile..." && \
#     echo "selected_scheme scheme-infraonly" > /tmp/texlive.profile && \
#     echo "tlpdbopt_install_docfiles 0" >> /tmp/texlive.profile && \
#     echo "tlpdbopt_install_srcfiles 0" >> /tmp/texlive.profile && \
#     echo "TEXDIR /usr/local/texlive" >> /tmp/texlive.profile && \
#     echo "TEXMFCONFIG /usr/local/texlive/texmf-config" >> /tmp/texlive.profile && \
#     echo "TEXMFHOME /usr/local/texlive/texmf-home" >> /tmp/texlive.profile && \
#     echo "TEXMFLOCAL /usr/local/texlive/texmf-local" >> /tmp/texlive.profile && \
#     echo "TEXMFSYSCONFIG /usr/local/texlive/texmf-config" >> /tmp/texlive.profile && \
#     echo "TEXMFSYSVAR /usr/local/texlive/texmf-var" >> /tmp/texlive.profile && \
#     echo "TEXMFVAR /usr/local/texlive/texmf-var" >> /tmp/texlive.profile && \
#     \
#     echo "🔄 Installing TeX Live base system..." && \
#     /tmp/install-tl-*/install-tl --profile=/tmp/texlive.profile && \
#     \
#     echo "🔧 Setting up TeX Live PATH..." && \
#     echo 'export PATH=/usr/local/texlive/bin/x86_64-linux:$PATH' >> /etc/bash.bashrc && \
#     export PATH=/usr/local/texlive/bin/x86_64-linux:$PATH && \
#     \
#     echo "📊 Analyzing tl_packages file..." && \
#     collection_count=$(grep -c '^collection-' /tmp/tl_packages) && \
#     echo "📦 Found $collection_count TeX Live collections to install" && \
#     \
#     echo "🔄 Installing TeX Live collections..." && \
#     export PATH=/usr/local/texlive/bin/x86_64-linux:$PATH && \
#     echo "📍 Checking if tlmgr is available..." && \
#     which tlmgr && tlmgr --version || echo "❌ tlmgr not found or not working" && \
#     i=1 && \
#     while IFS= read -r collection; do \
#         case "$collection" in \
#             collection-*) \
#                 echo "📦 [$i/$collection_count] Installing $collection..."; \
#                 if command -v tlmgr >/dev/null 2>&1; then \
#                     tlmgr install "$collection" || echo "⚠️ Failed to install $collection, continuing..."; \
#                 else \
#                     echo "⚠️ tlmgr not available, skipping $collection"; \
#                 fi; \
#                 i=$((i+1)); \
#                 ;; \
#         esac; \
#     done < /tmp/tl_packages && \
#     \
#     echo "🧹 Cleaning up TeX Live installer..." && \
#     rm -rf /tmp/install-tl-* /tmp/texlive.profile /tmp/install-tl-unx.tar.gz && \
#     \
#     end_time=$(date +%s) && \
#     duration=$((end_time - start_time)) && \
#     echo "✅ === TEX LIVE INSTALLATION COMPLETE === (${duration}s)" && \
#     echo "📊 Free disk space after: $(df -h / | tail -1 | awk '{print $4}')" && \
#     echo "📊 TeX Live disk usage: $(du -sh /usr/local/texlive 2>/dev/null || echo 'N/A')"

# Verify TeX Live installation (with error handling) (COMMENTED OUT FOR TESTING)
# RUN echo "🔄 Verifying TeX Live installation..." && \
#     export PATH=/usr/local/texlive/bin/x86_64-linux:$PATH && \
#     echo "📍 PATH: $PATH" && \
#     echo "📍 Checking TeX Live directory:" && \
#     ls -la /usr/local/texlive/ || echo "❌ TeX Live directory not found" && \
#     ls -la /usr/local/texlive/bin/ || echo "❌ TeX Live bin directory not found" && \
#     if [ -f /usr/local/texlive/bin/x86_64-linux/lualatex ]; then \
#         echo "✅ lualatex found"; \
#         /usr/local/texlive/bin/x86_64-linux/lualatex --version | head -1; \
#     else \
#         echo "❌ lualatex not found, checking for alternative locations"; \
#         find /usr/local/texlive -name "lualatex" -type f 2>/dev/null || echo "No lualatex found anywhere"; \
#     fi && \
#     echo "✅ TeX Live verification complete (allowing partial failures)"

# === PHASE 6: R INSTALLATION === (COMMENTED OUT FOR TESTING)
# RUN echo "🚀 === STARTING R INSTALLATION ===" && \
#     echo "⏰ Estimated time: 1-2 minutes" && \
#     start_time=$(date +%s) && \
#     \
#     echo "📦 Installing R and development packages..." && \
#     apt-get update && apt-get install -y \
#     r-base \
#     r-base-dev \
#     r-recommended && \
#     \
#     echo "🧹 Cleaning package cache..." && \
#     rm -rf /var/lib/apt/lists/* && \
#     \
#     echo "📊 R version: $(R --version | head -1)" && \
#     \
#     end_time=$(date +%s) && \
#     duration=$((end_time - start_time)) && \
#     echo "✅ === R INSTALLATION COMPLETE === (${duration}s)"

# === PHASE 7: PYTHON INSTALLATION === (COMMENTED OUT FOR TESTING)
# RUN echo "🚀 === STARTING PYTHON INSTALLATION ===" && \
#     echo "⏰ Estimated time: 1 minute" && \
#     start_time=$(date +%s) && \
#     \
#     echo "📦 Installing Python 3 and development packages..." && \
#     apt-get update && apt-get install -y \
#     python3 \
#     python3-pip \
#     python3-dev && \
#     \
#     echo "🧹 Cleaning package cache..." && \
#     rm -rf /var/lib/apt/lists/* && \
#     \
#     echo "📊 Python version: $(python3 --version)" && \
#     echo "📊 Pip version: $(pip3 --version)" && \
#     \
#     end_time=$(date +%s) && \
#     duration=$((end_time - start_time)) && \
#     echo "✅ === PYTHON INSTALLATION COMPLETE === (${duration}s)"

# === PHASE 3: QUARTO INSTALLATION ===
RUN echo "🚀 === STARTING QUARTO INSTALLATION ===" && \
    echo "⏰ Estimated time: 1 minute" && \
    start_time=$(date +%s) && \
    \
    echo "📦 Downloading Quarto 1.7.31..." && \
    wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.31/quarto-1.7.31-linux-amd64.deb && \
    \
    echo "📦 Installing Quarto..." && \
    dpkg -i quarto-1.7.31-linux-amd64.deb && \
    \
    echo "🧹 Cleaning up installer..." && \
    rm quarto-1.7.31-linux-amd64.deb && \
    \
    end_time=$(date +%s) && \
    duration=$((end_time - start_time)) && \
    echo "✅ === QUARTO INSTALLATION COMPLETE === (${duration}s)"

# Create R library directory (COMMENTED OUT FOR TESTING)
# RUN mkdir -p $R_LIBS_USER

# Copy dependency files (COMMENTED OUT FOR TESTING)
# COPY tools/dependencies/requirements/ /tmp/requirements/
# COPY tools/dependencies/requirements-build.txt /tmp/requirements.txt
# COPY tools/dependencies/install_packages.R /tmp/install_packages.R
# COPY docker/quarto-linux-build/verify_r_packages.R /tmp/verify_r_packages.R

# === PHASE 8: PYTHON PACKAGES === (COMMENTED OUT FOR TESTING)
# RUN echo "🚀 === STARTING PYTHON PACKAGE INSTALLATION ===" && \
#     echo "⏰ Estimated time: 1-2 minutes" && \
#     start_time=$(date +%s) && \
#     \
#     echo "🔄 Upgrading pip..." && \
#     pip3 install --upgrade pip && \
#     \
#     echo "📊 Analyzing requirements.txt..." && \
#     package_count=$(grep -v '^#' /tmp/requirements.txt | grep -v '^$' | wc -l) && \
#     echo "📦 Found $package_count Python packages to install" && \
#     \
#     echo "🔄 Installing Python packages with space optimization..." && \
#     pip3 install --no-cache-dir -r /tmp/requirements.txt && \
#     \
#     echo "🧹 Cleaning Python installation caches..." && \
#     pip3 cache purge && \
#     find /usr -name "*.pyc" -delete && \
#     find /usr -name "__pycache__" -type d -exec rm -rf {} + || true && \
#     \
#     echo "📊 Installed Python packages:" && \
#     pip3 list | head -10 && \
#     echo "📊 Total packages: $(pip3 list | wc -l)" && \
#     \
#     end_time=$(date +%s) && \
#     duration=$((end_time - start_time)) && \
#     echo "✅ === PYTHON PACKAGES COMPLETE === (${duration}s)"

# === PHASE 9: R PACKAGES === (COMMENTED OUT FOR TESTING)
# Install R packages (using same process as successful workflow)
# RUN R --slave -e " \
#     # Set options for better package installation \
#     options(repos = c(CRAN = 'https://cran.rstudio.com')); \
#     \
#     cat('🔄 Installing R packages...\n'); \
#     cat(paste('R library path:', Sys.getenv('R_LIBS_USER'), '\n')); \
#     \
#     # Create and set library path \
#     lib_path <- Sys.getenv('R_LIBS_USER'); \
#     dir.create(lib_path, showWarnings = FALSE, recursive = TRUE); \
#     .libPaths(lib_path); \
#     \
#     # Install packages \
#     cat('📦 Installing remotes package...\n'); \
#     install.packages('remotes'); \
#     \
#     if (file.exists('/tmp/install_packages.R')) { \
#       cat('📦 Installing packages from tools/dependencies/install_packages.R...\n'); \
#       source('/tmp/install_packages.R'); \
#     } else { \
#       cat('⚠️ No tools/dependencies/install_packages.R found, installing common packages\n'); \
#       pkgs <- c('rmarkdown', 'knitr', 'tidyverse', 'ggplot2', 'bookdown'); \
#       cat(paste('📦 Installing packages:', paste(pkgs, collapse=', '), '\n')); \
#       install.packages(pkgs); \
#     }; \
#     \
#     cat('✅ R package installation complete\n'); \
#     cat('📊 Installed packages:\n'); \
#     ip <- installed.packages()[, 'Package']; \
#     print(head(ip, 10)); \
#     cat(paste('Total packages installed:', nrow(ip), '\n')); \
#     "

# === PHASE 10: R PACKAGE VERIFICATION === (COMMENTED OUT FOR TESTING)
# RUN echo "🔍 Verifying R package installation..." && \
#     Rscript /tmp/verify_r_packages.R

# === PHASE 11: COMPREHENSIVE CLEANUP === (COMMENTED OUT FOR TESTING)
# RUN echo "🚀 === STARTING COMPREHENSIVE CLEANUP ===" && \
#     echo "📊 Disk space before cleanup: $(df -h / | tail -1 | awk '{print $4}')" && \
#     start_time=$(date +%s) && \
#     \
#     echo "🧹 Removing temporary files..." && \
#     rm -rf /tmp/* && \
#     rm -rf /var/tmp/* && \
#     \
#     echo "🧹 Cleaning package caches..." && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/* && \
#     rm -rf /var/cache/apt/* && \
#     \
#     echo "🧹 Cleaning Python caches..." && \
#     find /usr -name "*.pyc" -delete && \
#     find /usr -name "__pycache__" -type d -exec rm -rf {} + || true && \
#     pip3 cache purge || true && \
#     \
#     echo "🧹 Cleaning R temporary files..." && \
#     rm -rf /tmp/Rtmp* || true && \
#     rm -rf /var/lib/R/site-library/*/help || true && \
#     \
#     echo "🧹 Cleaning TeX Live caches and docs..." && \
#     rm -rf /usr/local/texlive/*/texmf-var/luatex-cache/* || true && \
#     rm -rf /usr/local/texlive/*/texmf-var/web2c/* || true && \
#     \
#     echo "🧹 Removing unnecessary system files..." && \
#     rm -rf /usr/share/doc/* && \
#     rm -rf /usr/share/man/* && \
#     rm -rf /usr/share/info/* && \
#     rm -rf /var/log/* && \
#     \
#     end_time=$(date +%s) && \
#     duration=$((end_time - start_time)) && \
#     echo "✅ === COMPREHENSIVE CLEANUP COMPLETE === (${duration}s)" && \
#     echo "📊 Final disk space: $(df -h / | tail -1 | awk '{print $4}')"

# Set working directory
WORKDIR /workspace

# Verify installations (MINIMAL TEST - JUST QUARTO)
RUN quarto --version && \
    echo "✅ Minimal test container - only Quarto installed"

# Health check (MINIMAL TEST VERSION)
RUN echo "✅ Minimal test container build completed successfully" && \
    echo "📊 Quarto version: $(quarto --version)" && \
    echo "🧪 This is a minimal test to verify build/save/find process works" && \
    echo "🚀 Ready to test container build workflow!"