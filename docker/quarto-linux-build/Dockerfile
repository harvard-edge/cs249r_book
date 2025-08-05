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

# Install system dependencies
RUN apt-get update && apt-get install -y \
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
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Inkscape from PPA
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:inkscape.dev/stable -y && \
    apt-get update && \
    apt-get install -y inkscape && \
    rm -rf /var/lib/apt/lists/*

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

# Install TeX Live using the same method as GitHub workflows for consistency
# Download and install TeX Live based on the tl_packages file
COPY tools/dependencies/tl_packages /tmp/
RUN apt-get update && apt-get install -y \
    perl \
    wget \
    xzdec && \
    rm -rf /var/lib/apt/lists/* && \
    # Install TeX Live installer
    wget -O /tmp/install-tl-unx.tar.gz "https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz" && \
    cd /tmp && tar -xzf install-tl-unx.tar.gz && \
    # Create TeX Live profile for automated installation
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
    # Install TeX Live with minimal scheme
    /tmp/install-tl-*/install-tl --profile=/tmp/texlive.profile && \
    # Add TeX Live to PATH
    echo 'export PATH=/usr/local/texlive/bin/x86_64-linux:$PATH' >> /etc/bash.bashrc && \
    export PATH=/usr/local/texlive/bin/x86_64-linux:$PATH && \
    # Install packages from tl_packages file
    grep -E '^collection-' /tmp/tl_packages | while read collection; do \
        echo "Installing $collection..."; \
        tlmgr install "$collection"; \
    done && \
    # Clean up
    rm -rf /tmp/install-tl-* /tmp/texlive.profile /tmp/install-tl-unx.tar.gz

# Verify TeX Live installation (simplified)
RUN export PATH=/usr/local/texlive/bin/x86_64-linux:$PATH && \
    echo "🔄 Verifying TeX Live installation..." && \
    which lualatex && which pdflatex && \
    lualatex --version | head -1 && \
    echo "✅ TeX Live verification successful"

# Install R
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    r-recommended && \
    rm -rf /var/lib/apt/lists/*

# Install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

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