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

# Install font dependencies
RUN apt-get update && apt-get install -y \
    fonts-freefont-ttf \
    fonts-liberation \
    fontconfig && \
    fc-cache -fv && \
    rm -rf /var/lib/apt/lists/*

# Install Ghostscript
RUN apt-get update && apt-get install -y ghostscript && rm -rf /var/lib/apt/lists/*

# Install TeX Live (full distribution for consistency)
RUN apt-get update && apt-get install -y \
    texlive-full \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-luatex \
    texlive-pictures && \
    rm -rf /var/lib/apt/lists/*

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

# Install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install -r /tmp/requirements.txt

# Install R packages
RUN Rscript /tmp/install_packages.R

# Clean up
RUN rm -rf /tmp/requirements.txt /tmp/install_packages.R

# Set working directory
WORKDIR /workspace

# Verify installations
RUN quarto --version && \
    python3 --version && \
    R --version && \
    lualatex --version

# Health check
RUN echo "✅ Container build completed successfully" && \
    echo "📊 Quarto version: $(quarto --version)" && \
    echo "📊 Python version: $(python3 --version)" && \
    echo "📊 R version: $(R --version | head -1)" && \
    echo "📊 TeX Live: $(lualatex --version | head -1)" 