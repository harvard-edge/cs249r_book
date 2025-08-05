# MLSysBook Windows Quarto Build Container
# Based on Windows Server Core with all dependencies pre-installed
# This container eliminates the 30-45 minute setup time for Windows builds

FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Set environment variables
ENV R_LIBS_USER=C:\r-lib
ENV QUARTO_LOG_LEVEL=INFO
ENV PYTHONIOENCODING=utf-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install Chocolatey package manager
RUN powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"

# Install system dependencies
RUN choco install -y \
    python \
    r \
    ghostscript \
    inkscape \
    git \
    curl \
    wget

# Install TeX Live
RUN choco install -y miktex

# Install Quarto
RUN powershell -Command "Invoke-WebRequest -Uri 'https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.31/quarto-1.7.31-win.zip' -OutFile 'quarto.zip'"
RUN powershell -Command "Expand-Archive -Path 'quarto.zip' -DestinationPath 'C:\quarto' -Force"
RUN setx PATH "%PATH%;C:\quarto\quarto-1.7.31\bin"

# Create R library directory
RUN mkdir C:\r-lib

# Copy dependency files
COPY tools/dependencies/requirements.txt C:\temp\
COPY tools/dependencies/install_packages.R C:\temp\

# Install Python packages
RUN python -m pip install --upgrade pip
RUN python -m pip install -r C:\temp\requirements.txt

# Install R packages
RUN Rscript C:\temp\install_packages.R

# Verify R package installation (same as Linux container)
RUN echo "🔍 Verifying R package installation..." && \
    Rscript -e "source('C:/temp/install_packages.R'); missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]; if(length(missing_packages) > 0) { cat('❌ Missing packages:', paste(missing_packages, collapse = ', '), '\n'); quit(status = 1) } else { cat('✅ All required R packages installed successfully\n') }"

# Clean up
RUN del C:\temp\requirements.txt C:\temp\install_packages.R

# Set working directory
WORKDIR C:\workspace

# Verify installations
RUN quarto --version
RUN python --version
RUN R --version
RUN lualatex --version

# Health check
RUN echo "✅ Windows container build completed successfully"
RUN echo "📊 Quarto version:"
RUN quarto --version
RUN echo "📊 Python version:"
RUN python --version
RUN echo "📊 R version:"
RUN R --version
RUN echo "📊 TeX Live:"
RUN lualatex --version 