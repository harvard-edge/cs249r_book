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
ENV PATH=C:\texlive\bin\windows;C:\quarto\quarto-1.7.31\bin;${PATH}

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

# Install TeX Live using same packages as workflows (instead of MikTeX)
# Note: Using TeX Live installer for Windows to match Linux container
COPY tools/dependencies/tl_packages C:\temp\
RUN powershell -Command " \
    # Download TeX Live installer for Windows \
    Invoke-WebRequest -Uri 'https://mirror.ctan.org/systems/texlive/tlnet/install-tl-windows.exe' -OutFile 'C:\temp\install-tl-windows.exe'; \
    # Create installation profile \
    @' \
selected_scheme scheme-infraonly \
tlpdbopt_install_docfiles 0 \
tlpdbopt_install_srcfiles 0 \
TEXDIR C:/texlive \
TEXMFCONFIG C:/texlive/texmf-config \
TEXMFHOME C:/texlive/texmf-home \
TEXMFLOCAL C:/texlive/texmf-local \
TEXMFSYSCONFIG C:/texlive/texmf-config \
TEXMFSYSVAR C:/texlive/texmf-var \
TEXMFVAR C:/texlive/texmf-var \
'@ | Out-File -FilePath 'C:\temp\texlive.profile' -Encoding ASCII; \
    # Run TeX Live installer \
    Start-Process -FilePath 'C:\temp\install-tl-windows.exe' -ArgumentList '-profile', 'C:\temp\texlive.profile' -Wait; \
    # Add TeX Live to PATH \
    $env:PATH += ';C:\texlive\bin\windows'; \
    [Environment]::SetEnvironmentVariable('PATH', $env:PATH, [EnvironmentVariableTarget]::Machine); \
    # Install collections from tl_packages \
    Get-Content 'C:\temp\tl_packages' | Where-Object { $_ -match '^collection-' } | ForEach-Object { \
        Write-Host \"Installing $_...\"; \
        tlmgr install $_; \
    }; \
    # Clean up \
    Remove-Item 'C:\temp\install-tl-windows.exe'; \
    Remove-Item 'C:\temp\texlive.profile' \
    "

# Install Quarto
RUN powershell -Command "Invoke-WebRequest -Uri 'https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.31/quarto-1.7.31-win.zip' -OutFile 'quarto.zip'"
RUN powershell -Command "Expand-Archive -Path 'quarto.zip' -DestinationPath 'C:\quarto' -Force"
RUN powershell -Command "$env:PATH += ';C:\quarto\quarto-1.7.31\bin'; [Environment]::SetEnvironmentVariable('PATH', $env:PATH, [EnvironmentVariableTarget]::Machine)"

# Create R library directory
RUN mkdir C:\r-lib

# Copy dependency files
COPY tools/dependencies/requirements.txt C:\temp\
COPY tools/dependencies/install_packages.R C:\temp\
COPY docker/quarto-build-windows/verify_r_packages.R C:\temp\

# Install Python packages
RUN python -m pip install --upgrade pip
RUN python -m pip install -r C:\temp\requirements.txt

# Install R packages (using same process as successful workflow)
RUN powershell -Command " \
    Rscript -e \" \
    # Set options for better package installation; \
    options(repos = c(CRAN = 'https://cran.rstudio.com')); \
    \
    cat('🔄 Installing R packages...\n'); \
    cat(paste('R library path:', Sys.getenv('R_LIBS_USER'), '\n')); \
    \
    # Create and set library path; \
    lib_path <- Sys.getenv('R_LIBS_USER'); \
    dir.create(lib_path, showWarnings = FALSE, recursive = TRUE); \
    .libPaths(lib_path); \
    \
    # Install packages; \
    cat('📦 Installing remotes package...\n'); \
    install.packages('remotes'); \
    \
    if (file.exists('C:/temp/install_packages.R')) { \
      cat('📦 Installing packages from tools/dependencies/install_packages.R...\n'); \
      source('C:/temp/install_packages.R'); \
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
    \" \
    "

# Verify R package installation (same as Linux container)
RUN powershell -Command "Write-Host '🔍 Verifying R package installation...'"
RUN Rscript C:\temp\verify_r_packages.R

# Clean up
RUN del C:\temp\requirements.txt C:\temp\install_packages.R C:\temp\verify_r_packages.R C:\temp\tl_packages

# Set working directory
WORKDIR C:\workspace

# Verify installations
RUN quarto --version
RUN python --version
RUN R --version
RUN lualatex --version

# Health check
RUN powershell -Command "Write-Host '✅ Windows container build completed successfully'" && \
    powershell -Command "Write-Host '📊 Quarto version: ' -NoNewline; quarto --version" && \
    powershell -Command "Write-Host '📊 Python version: ' -NoNewline; python --version" && \
    powershell -Command "Write-Host '📊 R version: ' -NoNewline; R --version" && \
    powershell -Command "Write-Host '📊 TeX Live: ' -NoNewline; lualatex --version" 