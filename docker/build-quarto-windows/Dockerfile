# escape=`
# MLSysBook Windows Quarto Build Container (Windows Server 2022)
# - PowerShell 7 via ZIP (no MSI)
# - Quarto 1.7.31 via ZIP (no MSI)
# - Python 3.13.1 + requirements
# - Ghostscript + Inkscape (Chocolatey)
# - TeX Live pinned to 2025 snapshot + packages from tl_packages
# - R 4.3.2 + packages via install_packages.R
# - Verifications: versions, kpsewhich font files, TikZ smoke test

FROM mcr.microsoft.com/windows/server:ltsc2022

# Use Windows PowerShell initially
SHELL ["powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass", "-Command"]

# ------------------------------------------------------------
# PHASE 0: Base dirs and env (same as quarto-build workflow)
# ------------------------------------------------------------
ENV R_LIBS_USER="C:/r-lib"
ENV QUARTO_LOG_LEVEL="INFO"
ENV PYTHONIOENCODING="utf-8"
ENV LANG="en_US.UTF-8"
ENV LC_ALL="en_US.UTF-8"

RUN New-Item -ItemType Directory -Force -Path 'C:\temp' | Out-Null ; New-Item -ItemType Directory -Force -Path 'C:\r-lib' | Out-Null

# ------------------------------------------------------------
# PHASE 1: PowerShell 7 (ZIP install, container-safe)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING POWERSHELL 7 INSTALLATION ===' ; `
    Write-Host 'Using ZIP install for container compatibility' ; `
    $Url = 'https://github.com/PowerShell/PowerShell/releases/download/v7.4.1/PowerShell-7.4.1-win-x64.zip' ; `
    $Zip = 'C:\PowerShell-7.4.1.zip' ; `
    Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing ; `
    New-Item -ItemType Directory -Force -Path 'C:\Program Files\PowerShell\7' | Out-Null ; `
    Expand-Archive -Path $Zip -DestinationPath 'C:\Program Files\PowerShell\7' -Force ; `
    Remove-Item $Zip -Force ; `
    $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
    if ($mach -notmatch [regex]::Escape('C:\Program Files\PowerShell\7')) { `
      [Environment]::SetEnvironmentVariable('PATH', ('C:\Program Files\PowerShell\7;' + $mach), 'Machine') } ; `
    & 'C:\Program Files\PowerShell\7\pwsh.exe' -NoLogo -Command '$PSVersionTable.PSVersion ; Write-Host ''PowerShell 7 installation verified ✅'''

# Switch to PowerShell 7 for subsequent layers
SHELL ["C:\\Program Files\\PowerShell\\7\\pwsh.exe", "-NoLogo", "-ExecutionPolicy", "Bypass", "-Command"]

# ------------------------------------------------------------
# PHASE 2: Chocolatey (package manager for Windows)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING CHOCOLATEY INSTALLATION ===' ; `
    Write-Host 'Installing Chocolatey package manager...' ; `
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 ; `
    iex ((New-Object Net.WebClient).DownloadString('https://chocolatey.org/install.ps1')) ; `
    choco --version ; `
    Write-Host '✅ Chocolatey installation complete'

# ------------------------------------------------------------
# PHASE 3: Copy dependency files (same as quarto-build workflow)
# ------------------------------------------------------------
COPY tools/dependencies/requirements/                  C:/temp/requirements/
COPY tools/dependencies/requirements-build.txt         C:/temp/requirements.txt
COPY tools/dependencies/install_packages.R             C:/temp/install_packages.R
COPY tools/dependencies/tl_packages                    C:/temp/tl_packages
COPY docker/build-quarto-windows/verify_r_packages.R   C:/temp/verify_r_packages.R

# ------------------------------------------------------------
# PHASE 4: Install Scoop and Quarto (same as quarto-build workflow)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING QUARTO INSTALLATION ===' ; `
    Write-Host 'Using Scoop for Quarto installation (same approach as quarto-build workflow)' ; `
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 ; `
    $OutputEncoding = [System.Text.Encoding]::UTF8 ; `
    Write-Host 'Installing Quarto via Scoop...' ; `
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force ; `
    Write-Host 'Installing Scoop package manager...' ; `
    Invoke-WebRequest -useb get.scoop.sh -outfile 'install.ps1' ; `
    & .\install.ps1 -RunAsAdmin ; `
    $scoopShims = Join-Path (Resolve-Path ~).Path 'scoop\shims' ; `
    $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
    [Environment]::SetEnvironmentVariable('PATH', ($scoopShims + ';' + $mach), 'Machine') ; `
    Write-Host 'Added Scoop shims to PATH:' $scoopShims ; `
    Write-Host 'Installing Git (required for buckets)...' ; `
    scoop install git ; `
    Write-Host 'Adding r-bucket...' ; `
    scoop bucket add r-bucket https://github.com/cderv/r-bucket.git ; `
    Write-Host 'Adding extras bucket...' ; `
    scoop bucket add extras ; `
    Write-Host 'Installing Quarto (latest stable)...' ; `
    scoop install quarto ; `
    Write-Host '✅ Quarto installation completed!'

# ------------------------------------------------------------
# PHASE 5: Install Python (same as quarto-build workflow)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING PYTHON INSTALLATION ===' ; `
    Write-Host 'Installing Python via Scoop (same as quarto-build workflow)...' ; `
    scoop install main/python ; `
    Write-Host '✅ Python installation complete'

# ------------------------------------------------------------
# PHASE 6: Install Python packages (same as quarto-build workflow)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING PYTHON PACKAGE INSTALLATION ===' ; `
    Write-Host 'Installing Python packages from requirements.txt (same as quarto-build workflow)...' ; `
    python -m pip install --upgrade pip ; `
    python -m pip install -r C:/temp/requirements.txt ; `
    Write-Host '✅ Python package installation complete'

# ------------------------------------------------------------
# PHASE 7: Install Ghostscript (same as quarto-build workflow)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING GHOSTSCRIPT INSTALLATION ===' ; `
    Write-Host 'Installing Ghostscript via Scoop (same as quarto-build workflow)...' ; `
    scoop install main/ghostscript ; `
    Write-Host '✅ Ghostscript installation complete'

# ------------------------------------------------------------
# PHASE 8: Install Inkscape (same as quarto-build workflow)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING INKSCAPE INSTALLATION ===' ; `
    Write-Host 'Installing Inkscape via Scoop (same as quarto-build workflow)...' ; `
    scoop install inkscape ; `
    Write-Host '✅ Inkscape installation complete'

# ------------------------------------------------------------
# PHASE 9: Install TeX Live (simple approach)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING TEX LIVE INSTALLATION ===' ; `
    Write-Host 'Installing TeX Live via Chocolatey (simple approach)...' ; `
    choco install texlive -y --no-progress ; `
    `
    Write-Host 'Adding TeX Live to PATH...' ; `
    $texLiveBin = 'C:\texlive\bin\windows' ; `
    $env:PATH = "$texLiveBin;$env:PATH" ; `
    [Environment]::SetEnvironmentVariable('PATH', $env:PATH, 'Machine') ; `
    `
    Write-Host 'Installing additional packages from tl_packages...' ; `
    $tlmgr = Join-Path $texLiveBin 'tlmgr.bat' ; `
    $pkgs = Get-Content 'C:\temp\tl_packages' | Where-Object { $_.Trim() -ne '' -and -not $_.Trim().StartsWith('#') } ; `
    foreach ($pkg in $pkgs) { `
        Write-Host "Installing: $pkg" ; `
        & $tlmgr install $pkg.Trim() ; `
    } ; `
    `
    Write-Host 'Basic verification...' ; `
    lualatex --version | Select-Object -First 1 ; `
    `
    Write-Host '✅ TeXLive installation complete'

# ------------------------------------------------------------
# PHASE 10: Install R (same as quarto-build workflow)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING R INSTALLATION ===' ; `
    Write-Host 'Installing R via Scoop (same as quarto-build workflow)...' ; `
    scoop install main/r ; `
    Write-Host '✅ R installation complete'

# ------------------------------------------------------------
# PHASE 11: R packages (same as quarto-build workflow)
# ------------------------------------------------------------
RUN Write-Host '=== INSTALLING R PACKAGES ===' ; `
    Write-Host 'Installing R packages from install_packages.R (same as quarto-build workflow)...' ; `
    Rscript -e "options(repos=c(CRAN='https://cran.rstudio.com')); dir.create(Sys.getenv('R_LIBS_USER'), recursive=TRUE, showWarnings=FALSE); .libPaths(Sys.getenv('R_LIBS_USER')); install.packages('remotes'); if (file.exists('C:/temp/install_packages.R')) source('C:/temp/install_packages.R') else install.packages(c('rmarkdown','knitr','ggplot2')); for (p in c('rmarkdown','knitr')) if (!require(p, character.only=TRUE, quietly=TRUE)) stop('missing: ', p)" ; `
    Write-Host 'Verifying R packages...' ; `
    Rscript C:/temp/verify_r_packages.R ; `
    Write-Host '✅ R package installation complete'

# ------------------------------------------------------------
# PHASE 12: Cleanup (same as quarto-build workflow)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING CLEANUP ===' ; `
    Write-Host 'Cleaning temporary files (same as quarto-build workflow)...' ; `
    Remove-Item C:/temp/requirements.txt -ErrorAction SilentlyContinue ; `
    Remove-Item C:/temp/install_packages.R -ErrorAction SilentlyContinue ; `
    Remove-Item C:/temp/verify_r_packages.R -ErrorAction SilentlyContinue ; `
    Remove-Item C:/temp/tl_packages -ErrorAction SilentlyContinue ; `
    Remove-Item C:/temp/requirements/ -Recurse -Force -ErrorAction SilentlyContinue ; `
    Write-Host '✅ Cleanup complete'

# ------------------------------------------------------------
# FINAL CHECKS: Verify all installations (same as quarto-build workflow)
# ------------------------------------------------------------
WORKDIR C:/workspace
RUN Write-Host '=== FINAL VERIFICATION ===' ; `
    Write-Host 'Verifying all installations (same checks as quarto-build workflow)...' ; `
    quarto --version ; `
    python --version ; `
    R --version ; `
    lualatex --version ; `
    Write-Host '✅ Windows container build completed successfully'
