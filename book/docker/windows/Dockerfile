# escape=`
# MLSysBook Windows Quarto Build Container (Windows Server 2022)
# - PowerShell 7 via ZIP (no MSI)
# - Quarto via Scoop (extras bucket)
# - Python 3.13.1 + requirements
# - Ghostscript + Inkscape (Scoop)
# - TeX Live (latest by default; overridable via TEXLIVE_VERSION build arg) + packages from tl_packages
# - R 4.5.2 + packages via install_packages.R
# - Verifications: versions, kpsewhich font files, TikZ smoke test

FROM mcr.microsoft.com/windows/server:ltsc2022
ARG TEXLIVE_VERSION=latest
ENV TEXLIVE_VERSION=${TEXLIVE_VERSION}

# Use Windows PowerShell initially
SHELL ["powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass", "-Command"]

# === PATH CONFIGURATION ===
# Paths are hardcoded since they're stable and ARG scope causes issues in multi-layer builds

# ------------------------------------------------------------
# PHASE 0: Base dirs and env (same as quarto-build workflow)
# ------------------------------------------------------------
ENV R_LIBS_USER="C:/r-lib"
ENV QUARTO_LOG_LEVEL="INFO"
ENV PYTHONIOENCODING="utf-8"
ENV LANG="en_US.UTF-8"
ENV LC_ALL="en_US.UTF-8"
# Explicitly set TMP/TEMP so the TeX Live Perl installer always has a valid temp dir
ENV TMP="C:/temp"
ENV TEMP="C:/temp"

RUN Write-Host '=== STARTING BASE SETUP ===' ; `
    Write-Host 'Creating base directories...' ; `
    New-Item -ItemType Directory -Force -Path 'C:\temp' | Out-Null ; `
    Write-Host '📁 Created C:\temp' ; `
    New-Item -ItemType Directory -Force -Path 'C:\r-lib' | Out-Null ; `
    Write-Host '📁 Created C:\r-lib' ; `
    Write-Host 'Environment variables set:' ; `
    Write-Host "  R_LIBS_USER: $env:R_LIBS_USER" ; `
    Write-Host "  QUARTO_LOG_LEVEL: $env:QUARTO_LOG_LEVEL" ; `
    Write-Host "  PYTHONIOENCODING: $env:PYTHONIOENCODING" ; `
    Write-Host "  LANG: $env:LANG" ; `
    Write-Host "  LC_ALL: $env:LC_ALL" ; `
    Write-Host '✅ Base setup complete'

# ------------------------------------------------------------
# PHASE 1: PowerShell 7 (ZIP install, container-safe)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING POWERSHELL 7 INSTALLATION ===' ; `
    Write-Host 'Using ZIP install for container compatibility' ; `
    Write-Host 'Download URL: https://github.com/PowerShell/PowerShell/releases/download/v7.4.1/PowerShell-7.4.1-win-x64.zip' ; `
    $Url = 'https://github.com/PowerShell/PowerShell/releases/download/v7.4.1/PowerShell-7.4.1-win-x64.zip' ; `
    $Zip = 'C:\PowerShell-7.4.1.zip' ; `
    Write-Host "Downloading PowerShell 7 to: $Zip" ; `
    Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing ; `
    Write-Host '📥 Download completed' ; `
    Write-Host 'Creating PowerShell directory...' ; `
    New-Item -ItemType Directory -Force -Path 'C:\Program Files\PowerShell\7' | Out-Null ; `
    Write-Host '📁 Directory created' ; `
    Write-Host 'Extracting ZIP file...' ; `
    Expand-Archive -Path $Zip -DestinationPath 'C:\Program Files\PowerShell\7' -Force ; `
    Write-Host '📦 Extraction completed' ; `
    Write-Host 'Cleaning up ZIP file...' ; `
    Remove-Item $Zip -Force ; `
    Write-Host '🧹 Cleanup completed' ; `
    Write-Host 'Adding PowerShell to PATH...' ; `
    $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
    Write-Host "Current PATH: $mach" ; `
    if ($mach -notmatch [regex]::Escape('C:\Program Files\PowerShell\7')) { `
      [Environment]::SetEnvironmentVariable('PATH', ('C:\Program Files\PowerShell\7;' + $mach), 'Machine') ; `
      Write-Host '🔗 PowerShell added to PATH' ; `
    } else { `
      Write-Host '⚠️ PowerShell already in PATH' ; `
    } ; `
    Write-Host 'Verifying PowerShell installation...' ; `
    & 'C:\Program Files\PowerShell\7\pwsh.exe' -NoLogo -Command '$PSVersionTable.PSVersion ; Write-Host ''PowerShell 7 installation verified ✅'''

# Switch to PowerShell 7 for subsequent layers
SHELL ["C:\\Program Files\\PowerShell\\7\\pwsh.exe", "-NoLogo", "-ExecutionPolicy", "Bypass", "-Command"]

# ------------------------------------------------------------
# PHASE 2: Chocolatey (package manager for Windows)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING CHOCOLATEY INSTALLATION ===' ; `
    Write-Host 'Installing Chocolatey package manager...' ; `
    Write-Host 'Setting TLS 1.2 for download...' ; `
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 ; `
    Write-Host '🔒 TLS 1.2 enabled' ; `
    Write-Host 'Downloading and executing Chocolatey install script...' ; `
    iex ((New-Object Net.WebClient).DownloadString('https://chocolatey.org/install.ps1')) ; `
    Write-Host '📦 Chocolatey install script executed' ; `
    Write-Host 'Verifying Chocolatey installation...' ; `
    choco --version ; `
    Write-Host '🔧 Configuring Chocolatey for CI stability...' ; `
    choco config set --name commandExecutionTimeoutSeconds --value 14400 ; `
    choco feature disable --name showDownloadProgress ; `
    Write-Host '✅ Chocolatey timeout/progress settings applied' ; `
    Write-Host '✅ Chocolatey installation complete'

# ------------------------------------------------------------
# PHASE 3: Copy dependency files (same as quarto-build workflow)
# ------------------------------------------------------------
COPY book/tools/dependencies/requirements.txt C:/temp/requirements.txt
COPY book/tools/dependencies/install_packages.R C:/temp/install_packages.R
COPY book/tools/dependencies/tl_packages C:/temp/tl_packages
COPY book/docker/windows/verify_r_packages.R C:/temp/verify_r_packages.R
COPY book/docker/windows/install_texlive.ps1 C:/temp/install_texlive.ps1
RUN Write-Host '✅ Dependency file copy complete'

# ------------------------------------------------------------
# PHASE 4: Install Scoop (Package manager setup)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING SCOOP INSTALLATION ===' ; `
    Write-Host 'Setting UTF-8 encoding...' ; `
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 ; `
    $OutputEncoding = [System.Text.Encoding]::UTF8 ; `
    Write-Host '🔤 UTF-8 encoding set' ; `
    Write-Host 'Setting execution policy...' ; `
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force ; `
    Write-Host '🔐 Execution policy set' ; `
    Write-Host 'Installing Scoop package manager...' ; `
    Invoke-WebRequest -useb get.scoop.sh -outfile 'install.ps1' ; `
    Write-Host '📥 Scoop install script downloaded' ; `
    & .\install.ps1 -RunAsAdmin ; `
    Write-Host '📦 Scoop installed' ; `
    Write-Host 'Adding Scoop shims to PATH...' ; `
    $scoopShims = Join-Path (Resolve-Path ~).Path 'scoop\shims' ; `
    Write-Host "Scoop shims path: $scoopShims" ; `
    $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
    [Environment]::SetEnvironmentVariable('PATH', ($scoopShims + ';' + $mach), 'Machine') ; `
    Write-Host '🔗 Added Scoop shims to PATH' ; `
    Write-Host 'Installing Git (required for buckets)...' ; `
    scoop install git ; `
    Write-Host '📦 Git installed' ; `
    Write-Host 'Adding r-bucket...' ; `
    scoop bucket add r-bucket https://github.com/cderv/r-bucket.git ; `
    Write-Host '📦 r-bucket added' ; `
    Write-Host 'Adding extras bucket...' ; `
    scoop bucket add extras ; `
    Write-Host '📦 extras bucket added' ; `
    Write-Host '✅ Scoop installation completed!'

# ------------------------------------------------------------
# PHASE 5: Install Quarto via Scoop (consistent PATH with other tools)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING QUARTO INSTALLATION ===' ; `
    Write-Host 'Installing Quarto via Scoop (extras bucket)...' ; `
    scoop install extras/quarto ; `
    if (-not (Get-Command quarto -ErrorAction SilentlyContinue)) { `
        Write-Host '❌ Quarto install failed: executable not found in PATH' ; `
        exit 1 ; `
    } ; `
    $qv = quarto --version ; `
    if ($LASTEXITCODE -ne 0) { `
        Write-Host "❌ Quarto install failed: version check exited $LASTEXITCODE" ; `
        exit 1 ; `
    } ; `
    Write-Host "📦 Quarto installed: $qv" ; `
    Write-Host '✅ Quarto installation completed!'

# ------------------------------------------------------------
# PHASE 6: Install Ghostscript (required for PDF generation)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING GHOSTSCRIPT INSTALLATION ===' ; `
    Write-Host 'Installing Ghostscript via Scoop...' ; `
    scoop install main/ghostscript ; `
    if (-not (Get-Command gswin64c -ErrorAction SilentlyContinue)) { `
        Write-Host '❌ Ghostscript install failed: gswin64c not found in PATH' ; `
        exit 1 ; `
    } ; `
    $gsv = gswin64c --version ; `
    if ($LASTEXITCODE -ne 0) { `
        Write-Host "❌ Ghostscript install failed: version check exited $LASTEXITCODE" ; `
        exit 1 ; `
    } ; `
    Write-Host "📦 Ghostscript installed: $gsv" ; `
    Write-Host '✅ Ghostscript installation complete'

# ------------------------------------------------------------
# PHASE 7: Install Inkscape and rsvg-convert (required for SVG processing)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING INKSCAPE INSTALLATION ===' ; `
    Write-Host 'Installing Inkscape via Scoop...' ; `
    scoop install inkscape ; `
    if (-not (Get-Command inkscape -ErrorAction SilentlyContinue)) { `
        Write-Host '❌ Inkscape install failed: executable not found in PATH' ; `
        exit 1 ; `
    } ; `
    $iv = inkscape --version ; `
    if ($LASTEXITCODE -ne 0) { `
        Write-Host "❌ Inkscape install failed: version check exited $LASTEXITCODE" ; `
        exit 1 ; `
    } ; `
    Write-Host "📦 Inkscape installed: $iv" ; `
    Write-Host '✅ Inkscape installation complete'

RUN Write-Host '=== STARTING RSVG-CONVERT INSTALLATION ===' ; `
    Write-Host 'Installing rsvg-convert via Scoop (required by Quarto for SVG-to-PDF)...' ; `
    scoop install rsvg-convert ; `
    if (-not (Get-Command rsvg-convert -ErrorAction SilentlyContinue)) { `
        Write-Host '⚠️ Scoop install failed or package unavailable from current buckets; trying Chocolatey...' ; `
        choco install rsvg-convert -y ; `
    } ; `
    if (-not (Get-Command rsvg-convert -ErrorAction SilentlyContinue)) { `
        Write-Host '❌ rsvg-convert install failed via both Scoop and Chocolatey' ; `
        exit 1 ; `
    } ; `
    $rv = rsvg-convert --version ; `
    if ($LASTEXITCODE -ne 0) { `
        Write-Host "❌ rsvg-convert install failed: version check exited $LASTEXITCODE" ; `
        exit 1 ; `
    } ; `
    Write-Host "📦 rsvg-convert installed: $rv" ; `
    Write-Host '✅ rsvg-convert installation complete'

# ------------------------------------------------------------
# PHASE 8: Install Python (Medium complexity)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING PYTHON INSTALLATION ===' ; `
    Write-Host 'Installing Python via Scoop (same as quarto-build workflow)...' ; `
    Write-Host 'Installing Python from main bucket...' ; `
    scoop install main/python ; `
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) { `
        Write-Host '❌ Python install failed: executable not found in PATH' ; `
        exit 1 ; `
    } ; `
    $pv = python --version ; `
    if ($LASTEXITCODE -ne 0) { `
        Write-Host "❌ Python install failed: version check exited $LASTEXITCODE" ; `
        exit 1 ; `
    } ; `
    Write-Host "📦 Python installed: $pv" ; `
    Write-Host '✅ Python installation complete'

# ------------------------------------------------------------
# PHASE 9: Install Python packages (Medium complexity)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING PYTHON PACKAGE INSTALLATION ===' ; `
    Write-Host 'Installing Python packages from requirements.txt (same as quarto-build workflow)...' ; `
    Write-Host 'Using bundled pip (skip upgrade to avoid WinError 3 shim lock)...' ; `
    Write-Host 'Installing packages from requirements.txt...' ; `
    Write-Host 'Requirements file contents:' ; `
    Get-Content C:/temp/requirements.txt | Write-Host ; `
    python -m pip install -r C:/temp/requirements.txt ; `
    Write-Host '✅ Python package installation complete'

# ------------------------------------------------------------
# PHASE 10: Install Visual C++ Redistributable (Required for Quarto DLLs)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING VISUAL C++ REDISTRIBUTABLE INSTALLATION ===' ; `
    Write-Host 'Installing Microsoft Visual C++ Redistributable...' ; `
    Write-Host 'This is required for Quarto DLL dependencies on Windows' ; `
    choco install vcredist-all -y ; `
    Write-Host '📦 Visual C++ Redistributable installed' ; `
    Write-Host '✅ Visual C++ Redistributable installation complete'

# ------------------------------------------------------------
# PHASE 11: Install R (Medium complexity)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING R INSTALLATION ===' ; `
    Write-Host 'Installing R 4.5.2 via Scoop (main bucket)...' ; `
    scoop install main/r@4.5.2 ; `
    if (-not (Get-Command Rscript -ErrorAction SilentlyContinue)) { `
        Write-Host '❌ R install failed: Rscript not found in PATH' ; `
        exit 1 ; `
    } ; `
    $rv = Rscript --version ; `
    if ($LASTEXITCODE -ne 0) { `
        Write-Host "❌ R install failed: version check exited $LASTEXITCODE" ; `
        exit 1 ; `
    } ; `
    Write-Host "📦 R installed: $rv" ; `
    Write-Host '✅ R installation complete'

# ------------------------------------------------------------
# PHASE 12: Install R packages (Medium complexity)
# ------------------------------------------------------------
RUN Write-Host '=== INSTALLING R PACKAGES ===' ; `
    Write-Host 'Installing R packages from install_packages.R (same as quarto-build workflow)...' ; `
    Write-Host 'Setting up R environment...' ; `
    Write-Host "R_LIBS_USER: $env:R_LIBS_USER" ; `
    Write-Host 'Installing R packages...' ; `
    Rscript -e 'options(repos=c(CRAN=\"https://cran.rstudio.com\"))' ; `
    Rscript -e 'dir.create(Sys.getenv(\"R_LIBS_USER\"), recursive=TRUE, showWarnings=FALSE)' ; `
    Rscript -e '.libPaths(Sys.getenv(\"R_LIBS_USER\"))' ; `
    Rscript -e 'install.packages(\"remotes\")' ; `
    if (Test-Path 'C:/temp/install_packages.R') { `
        Write-Host 'Found install_packages.R, sourcing it...' ; `
        Rscript 'C:/temp/install_packages.R' ; `
    } else { `
        Write-Host 'No install_packages.R found, installing basic packages...' ; `
        Rscript -e 'install.packages(c(\"rmarkdown\",\"knitr\",\"ggplot2\"))' ; `
    } ; `
    Rscript -e 'for (p in c(\"rmarkdown\",\"knitr\")) if (!require(p, character.only=TRUE, quietly=TRUE)) stop(\"missing: \", p)' ; `
    Write-Host '📦 R packages installed' ; `
    Write-Host 'Verifying R packages...' ; `
    Rscript C:/temp/verify_r_packages.R ; `
    Write-Host '✅ R package installation complete'

# ------------------------------------------------------------
# PHASE 13: Install TeX Live LAST (largest/slowest install)
# Direct install-tl approach via standalone script:
# - Bypasses Chocolatey's ErrorActionPreference=Stop wrapper
# - Pins mirror in profile, tries 3 mirrors in order
# ------------------------------------------------------------
RUN & 'C:\temp\install_texlive.ps1'

RUN Write-Host '=== VERIFY TEX LIVE INSTALLATION ===' ; `
    $cmd = Get-Command lualatex -ErrorAction SilentlyContinue ; `
    if (-not $cmd) { `
        Write-Host '❌ TeX Live verification failed: lualatex not found in PATH' ; `
        exit 1 ; `
    } ; `
    Write-Host "Resolved: $($cmd.Source)" ; `
    $lv = lualatex --version | Select-Object -First 1 ; `
    if ($LASTEXITCODE -ne 0) { `
        Write-Host "❌ TeX Live verification failed: lualatex --version exited $LASTEXITCODE" ; `
        exit 1 ; `
    } ; `
    Write-Host "📦 LaTeX verified: $lv" ; `
    Write-Host '✅ TeX Live verification complete'

# ------------------------------------------------------------
# PHASE 14: Cleanup and Environment Setup
# ------------------------------------------------------------
RUN Write-Host '=== STARTING CLEANUP AND ENVIRONMENT SETUP ===' ; `
    Write-Host 'Cleaning temporary files and setting up environment...' ; `
    Write-Host 'Removing temporary files...' ; `
    Remove-Item C:/temp/requirements.txt -ErrorAction SilentlyContinue ; `
    Write-Host '🗑️ requirements.txt removed' ; `
    Remove-Item C:/temp/install_packages.R -ErrorAction SilentlyContinue ; `
    Write-Host '🗑️ install_packages.R removed' ; `
    Remove-Item C:/temp/verify_r_packages.R -ErrorAction SilentlyContinue ; `
    Write-Host '🗑️ verify_r_packages.R removed' ; `
    Remove-Item C:/temp/tl_packages -ErrorAction SilentlyContinue ; `
    Write-Host '🗑️ tl_packages removed' ; `
    Remove-Item C:/temp/requirements/ -Recurse -Force -ErrorAction SilentlyContinue ; `
    Write-Host '🗑️ requirements/ directory removed' ; `
    Write-Host 'Setting up environment variables for Quarto...' ; `
    $env:QUARTO_LOG_LEVEL = 'DEBUG' ; `
    [Environment]::SetEnvironmentVariable('QUARTO_LOG_LEVEL', 'DEBUG', 'Machine') ; `
    Write-Host '🔧 QUARTO_LOG_LEVEL set to DEBUG' ; `
    Write-Host '✅ Cleanup and environment setup complete'

# ------------------------------------------------------------
# PHASE 15: Commit PATH into Docker image metadata
# [Environment]::SetEnvironmentVariable writes to the registry but Docker does
# not re-read HKLM between RUN steps — each new layer inherits the *previous
# layer's* Docker ENV, not registry changes made mid-layer.  Using the Docker
# ENV directive here guarantees these paths are committed into the image and
# visible to every subsequent RUN step and to containers at runtime.
# ------------------------------------------------------------
ENV PATH="C:\\Users\\ContainerAdministrator\\scoop\\shims;\
C:\\Users\\ContainerAdministrator\\scoop\\apps\\python\\current\\Scripts;\
C:\\Users\\ContainerAdministrator\\scoop\\apps\\python\\current;\
C:\\Users\\ContainerAdministrator\\scoop\\apps\\ghostscript\\current\\bin;\
C:\\Users\\ContainerAdministrator\\scoop\\apps\\r\\current\\bin;\
C:\\Users\\ContainerAdministrator\\scoop\\apps\\git\\current\\cmd;\
C:\\texlive\\bin\\windows;\
C:\\Program Files\\PowerShell\\7;\
C:\\ProgramData\\chocolatey\\bin;\
C:\\Windows\\system32;\
C:\\Windows;\
C:\\Windows\\System32\\Wbem;\
C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\;\
C:\\Windows\\System32\\OpenSSH\\"

# ------------------------------------------------------------
# FINAL CHECKS: Comprehensive verification with diagnostics
# ------------------------------------------------------------
WORKDIR C:/workspace
RUN Write-Host '=== FINAL VERIFICATION ===' ; `
    Write-Host 'PATH environment variable:' ; `
    Write-Host $env:PATH ; `
    Write-Host '' ; `
    Write-Host 'Tool checks run in separate Docker steps below for isolated reporting'

RUN Write-Host '=== VERIFY: Quarto ===' ; `
    $cmd = Get-Command quarto -ErrorAction SilentlyContinue ; `
    if (-not $cmd) { Write-Host "❌ Quarto FAILED: command not found"; exit 1 } ; `
    Write-Host "Resolved: $($cmd.Source)" ; `
    quarto --version ; `
    Write-Host '✅ Quarto verified'

RUN Write-Host '=== VERIFY: Python ===' ; `
    $cmd = Get-Command python -ErrorAction SilentlyContinue ; `
    if (-not $cmd) { Write-Host "❌ Python FAILED: command not found"; exit 1 } ; `
    Write-Host "Resolved: $($cmd.Source)" ; `
    python --version ; `
    Write-Host '✅ Python verified'

RUN Write-Host '=== VERIFY: R ===' ; `
    $cmd = Get-Command Rscript -ErrorAction SilentlyContinue ; `
    if (-not $cmd) { Write-Host "❌ R FAILED: command not found"; exit 1 } ; `
    Write-Host "Resolved: $($cmd.Source)" ; `
    Rscript --version ; `
    Write-Host '✅ R verified'

RUN Write-Host '=== VERIFY: LaTeX ===' ; `
    $cmd = Get-Command lualatex -ErrorAction SilentlyContinue ; `
    if (-not $cmd) { Write-Host "❌ LaTeX FAILED: command not found"; exit 1 } ; `
    Write-Host "Resolved: $($cmd.Source)" ; `
    lualatex --version | Select-Object -First 1 ; `
    Write-Host '✅ LaTeX verified'

RUN Write-Host '=== VERIFY: Ghostscript ===' ; `
    $cmd = Get-Command gswin64c -ErrorAction SilentlyContinue ; `
    if (-not $cmd) { Write-Host "❌ Ghostscript FAILED: gswin64c not found"; exit 1 } ; `
    Write-Host "Resolved: $($cmd.Source)" ; `
    gswin64c --version ; `
    Write-Host '✅ Ghostscript verified'

RUN Write-Host '=== VERIFY: Inkscape ===' ; `
    $cmd = Get-Command inkscape -ErrorAction SilentlyContinue ; `
    if (-not $cmd) { Write-Host "❌ Inkscape FAILED: command not found"; exit 1 } ; `
    Write-Host "Resolved: $($cmd.Source)" ; `
    inkscape --version ; `
    Write-Host '✅ Inkscape verified'

RUN Write-Host '=== VERIFY: rsvg-convert ===' ; `
    $cmd = Get-Command rsvg-convert -ErrorAction SilentlyContinue ; `
    if (-not $cmd) { Write-Host "❌ rsvg-convert FAILED: command not found"; exit 1 } ; `
    Write-Host "Resolved: $($cmd.Source)" ; `
    rsvg-convert --version ; `
    Write-Host '✅ rsvg-convert verified'
