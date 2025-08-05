# MLSysBook Windows Quarto Build Container
# Replicates the exact environment that quarto-build workflow uses successfully
# Based on Windows Server 2022 (same as windows-latest GitHub Actions runner)

FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Install PowerShell 7 first using curl (available in Windows Server Core)
RUN curl -fSLo PowerShell.msi https://github.com/PowerShell/PowerShell/releases/download/v7.4.1/PowerShell-7.4.1-win-x64.msi && \
    msiexec /i PowerShell.msi /quiet /norestart && \
    del PowerShell.msi

# Install Chocolatey using PowerShell 7 (pwsh)
RUN pwsh -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"

# Set environment variables to match workflow
ENV R_LIBS_USER=C:\r-lib
ENV QUARTO_LOG_LEVEL=INFO
ENV PYTHONIOENCODING=utf-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Copy dependency files needed for installations
COPY tools/dependencies/requirements.txt C:\temp\
COPY tools/dependencies/install_packages.R C:\temp\
COPY tools/dependencies/tl_packages C:\temp\
COPY docker/quarto-build-windows/verify_r_packages.R C:\temp\

# Create R library directory
RUN mkdir C:\r-lib

# Setup Quarto (using exact version from workflow)
RUN pwsh -Command "Invoke-WebRequest -Uri 'https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.31/quarto-1.7.31-win.zip' -OutFile 'quarto.zip'" && \
    pwsh -Command "Expand-Archive -Path 'quarto.zip' -DestinationPath 'C:\quarto' -Force" && \
    pwsh -Command "del quarto.zip" && \
    setx PATH "%PATH%;C:\quarto\quarto-1.7.31\bin" /M

# Setup Python 3.13 (same as workflow)
RUN choco install -y python --version=3.13.1

# Install Python dependencies (same as workflow)
RUN python -m pip install --upgrade pip && \
    python -m pip install -r C:\temp\requirements.txt

# Install Ghostscript (using exact method from workflow)
RUN pwsh -Command " \
    try { \
        winget install ArtifexSoftware.GhostScript --accept-source-agreements --accept-package-agreements; \
        Write-Output '✅ Ghostscript installed via winget'; \
    } catch { \
        Write-Output '🔄 winget not available, using chocolatey...'; \
        choco install ghostscript -y; \
    } \
    $gsPath = Get-ChildItem 'C:\Program Files\gs' | Sort-Object Name -Descending | Select-Object -First 1; \
    $binPath = Join-Path $gsPath.FullName 'bin'; \
    Write-Output \"Adding Ghostscript path: $binPath\"; \
    [Environment]::SetEnvironmentVariable('PATH', \"$env:PATH;$binPath\", [EnvironmentVariableTarget]::Machine) \
    "

# Install Inkscape (using exact method from workflow)
RUN pwsh -Command " \
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; \
    $OutputEncoding = [System.Text.Encoding]::UTF8; \
    [Console]::InputEncoding = [System.Text.Encoding]::UTF8; \
    try { \
        winget install Inkscape.Inkscape --accept-source-agreements --accept-package-agreements; \
        Write-Output '✅ Inkscape installed via winget'; \
    } catch { \
        Write-Output '🔄 winget not available, using chocolatey...'; \
        choco install inkscape -y; \
    } \
    [Environment]::SetEnvironmentVariable('PATH', \"$env:PATH;C:\Program Files\Inkscape\bin\", [EnvironmentVariableTarget]::Machine) \
    "

# Install TeX Live packages (using zauguin/install-texlive equivalent)
# Note: This simulates what the GitHub Action does
RUN pwsh -Command " \
    # Download TeX Live installer \
    Invoke-WebRequest -Uri 'https://mirror.ctan.org/systems/texlive/tlnet/install-tl-windows.exe' -OutFile 'C:\temp\install-tl-windows.exe'; \
    # Create minimal installation profile \
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
    # Install TeX Live base \
    Start-Process -FilePath 'C:\temp\install-tl-windows.exe' -ArgumentList '-profile', 'C:\temp\texlive.profile' -Wait; \
    # Add TeX Live to PATH \
    [Environment]::SetEnvironmentVariable('PATH', \"$env:PATH;C:\texlive\bin\windows\", [EnvironmentVariableTarget]::Machine); \
    # Install collections from tl_packages file \
    $packages = Get-Content 'C:\temp\tl_packages' | Where-Object { $_.Trim() -ne '' }; \
    foreach ($package in $packages) { \
        Write-Host \"Installing TeX Live collection: $package\"; \
        & 'C:\texlive\bin\windows\tlmgr.bat' install $package.Trim(); \
    }; \
    # Clean up installer \
    Remove-Item 'C:\temp\install-tl-windows.exe'; \
    Remove-Item 'C:\temp\texlive.profile' \
    "

# Setup R (using exact version from workflow)
RUN choco install -y r --version=4.4.1

# Install R packages (using simplified workflow approach)
RUN Rscript -e " \
    options(repos = c(CRAN = 'https://cran.rstudio.com')); \
    cat('🔄 Installing R packages...\n'); \
    lib_path <- Sys.getenv('R_LIBS_USER'); \
    dir.create(lib_path, showWarnings = FALSE, recursive = TRUE); \
    .libPaths(lib_path); \
    install.packages('remotes'); \
    if (file.exists('C:/temp/install_packages.R')) { \
      source('C:/temp/install_packages.R'); \
    } \
    "

# Verify R package installation
RUN pwsh -Command "Write-Host '🔍 Verifying R package installation...'" && \
    Rscript C:\temp\verify_r_packages.R

# Clean up
RUN del C:\temp\requirements.txt C:\temp\install_packages.R C:\temp\verify_r_packages.R C:\temp\tl_packages

# Set working directory
WORKDIR C:\workspace

# Final verification (using same tools that workflow uses)
RUN quarto --version && \
    python --version && \
    R --version && \
    lualatex --version

# Health check
RUN pwsh -Command "Write-Host '✅ Windows container build completed successfully'"