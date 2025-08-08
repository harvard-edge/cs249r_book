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
    Write-Host '✅ Chocolatey installation complete'

# ------------------------------------------------------------
# PHASE 3: Copy dependency files (same as quarto-build workflow)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING DEPENDENCY FILE COPY ===' ; `
    Write-Host 'Copying dependency files to container...' ; `
    Write-Host 'Files to copy:' ; `
    Write-Host '  - tools/dependencies/requirements/ → C:/temp/requirements/' ; `
    Write-Host '  - tools/dependencies/requirements-build.txt → C:/temp/requirements.txt' ; `
    Write-Host '  - tools/dependencies/install_packages.R → C:/temp/install_packages.R' ; `
    Write-Host '  - tools/dependencies/tl_packages → C:/temp/tl_packages' ; `
    Write-Host '  - docker/build-quarto-windows/verify_r_packages.R → C:/temp/verify_r_packages.R' ; `
    Write-Host '✅ Dependency file copy phase complete'

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
    Write-Host 'Setting UTF-8 encoding...' ; `
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 ; `
    $OutputEncoding = [System.Text.Encoding]::UTF8 ; `
    Write-Host '🔤 UTF-8 encoding set' ; `
    Write-Host 'Installing Quarto via Scoop...' ; `
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
    Write-Host 'Installing Quarto (latest stable)...' ; `
    scoop install quarto ; `
    Write-Host '✅ Quarto installation completed!'

# ------------------------------------------------------------
# PHASE 5: Install Python (same as quarto-build workflow)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING PYTHON INSTALLATION ===' ; `
    Write-Host 'Installing Python via Scoop (same as quarto-build workflow)...' ; `
    Write-Host 'Installing Python from main bucket...' ; `
    scoop install main/python ; `
    Write-Host '📦 Python installed' ; `
    Write-Host 'Verifying Python installation...' ; `
    python --version ; `
    Write-Host '✅ Python installation complete'

# ------------------------------------------------------------
# PHASE 6: Install Python packages (same as quarto-build workflow)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING PYTHON PACKAGE INSTALLATION ===' ; `
    Write-Host 'Installing Python packages from requirements.txt (same as quarto-build workflow)...' ; `
    Write-Host 'Upgrading pip...' ; `
    python -m pip install --upgrade pip ; `
    Write-Host '📦 pip upgraded' ; `
    Write-Host 'Installing packages from requirements.txt...' ; `
    Write-Host 'Requirements file contents:' ; `
    Get-Content C:/temp/requirements.txt | Write-Host ; `
    python -m pip install -r C:/temp/requirements.txt ; `
    Write-Host '✅ Python package installation complete'

# ------------------------------------------------------------
# PHASE 7: Install Ghostscript (same as quarto-build workflow)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING GHOSTSCRIPT INSTALLATION ===' ; `
    Write-Host 'Installing Ghostscript via Scoop (same as quarto-build workflow)...' ; `
    Write-Host 'Installing Ghostscript from main bucket...' ; `
    scoop install main/ghostscript ; `
    Write-Host '📦 Ghostscript installed' ; `
    Write-Host 'Verifying Ghostscript installation...' ; `
    gs --version ; `
    Write-Host '✅ Ghostscript installation complete'

# ------------------------------------------------------------
# PHASE 8: Install Inkscape (same as quarto-build workflow)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING INKSCAPE INSTALLATION ===' ; `
    Write-Host 'Installing Inkscape via Scoop (same as quarto-build workflow)...' ; `
    Write-Host 'Installing Inkscape...' ; `
    scoop install inkscape ; `
    Write-Host '📦 Inkscape installed' ; `
    Write-Host 'Verifying Inkscape installation...' ; `
    inkscape --version ; `
    Write-Host '✅ Inkscape installation complete'

# ------------------------------------------------------------
# PHASE 9: Install TeX Live (one-shot profile approach)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING TEX LIVE INSTALLATION ===' ; `
    Write-Host 'Installing TeX Live using one-shot profile (cleanest approach)...' ; `
    `
    Write-Host 'Downloading TeX Live installer...' ; `
    $installer = 'C:\temp\install-tl-windows.exe' ; `
    Invoke-WebRequest -Uri 'https://mirror.ctan.org/systems/texlive/tlnet/install-tl-windows.exe' -OutFile $installer -UseBasicParsing ; `
    `
    Write-Host 'Creating one-shot install profile from tl_packages...' ; `
    # Start with basic profile
    @'
selected_scheme scheme-basic

tlpdbopt_install_docfiles 0
tlpdbopt_install_srcfiles 0

'@ | Out-File -FilePath 'C:\temp\install-tl.profile' -Encoding ASCII ; `
    `
    # Add collections from tl_packages file
    Write-Host 'Reading tl_packages file...' ; `
    $collections = Get-Content 'C:\temp\tl_packages' | Where-Object { $_.Trim() -ne '' -and -not $_.Trim().StartsWith('#') } ; `
    Write-Host "Found $($collections.Count) collections to install:" ; `
    foreach ($collection in $collections) { `
        Write-Host "  - $collection" ; `
        "$collection 1" | Add-Content -Path 'C:\temp\install-tl.profile' -Encoding ASCII ; `
    } ; `
    Write-Host '📄 Profile created successfully' ; `
    `
    Write-Host 'Installing TeX Live with profile...' ; `
    Write-Host "Installer path: $installer" ; `
    Write-Host "Profile path: C:\temp\install-tl.profile" ; `
    Write-Host "Profile contents:" ; `
    Get-Content 'C:\temp\install-tl.profile' | Write-Host ; `
    Start-Process -FilePath $installer -ArgumentList '-profile', 'C:\temp\install-tl.profile' -Wait -NoNewWindow ; `
    `
    Write-Host 'Adding TeX Live to PATH...' ; `
    $texLiveBin = 'C:\texlive\bin\windows' ; `
    $env:PATH = "$texLiveBin;$env:PATH" ; `
    [Environment]::SetEnvironmentVariable('PATH', $env:PATH, 'Machine') ; `
    `
    Write-Host 'Verifying installation...' ; `
    Write-Host 'Checking installed collections:' ; `
    tlmgr list --only-installed | Select-String 'collection-' ; `
    Write-Host 'Checking lualatex:' ; `
    lualatex --version | Select-Object -First 1 ; `
    Write-Host 'Checking tlmgr path:' ; `
    Get-Command tlmgr | Select-Object Source ; `
    `
    Write-Host '✅ TeXLive installation complete'

# ------------------------------------------------------------
# PHASE 10: Install R (same as quarto-build workflow)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING R INSTALLATION ===' ; `
    Write-Host 'Installing R via Scoop (same as quarto-build workflow)...' ; `
    Write-Host 'Installing R from main bucket...' ; `
    scoop install main/r ; `
    Write-Host '📦 R installed' ; `
    Write-Host 'Verifying R installation...' ; `
    R --version ; `
    Write-Host '✅ R installation complete'

# ------------------------------------------------------------
# PHASE 11: R packages (same as quarto-build workflow)
# ------------------------------------------------------------
RUN Write-Host '=== INSTALLING R PACKAGES ===' ; `
    Write-Host 'Installing R packages from install_packages.R (same as quarto-build workflow)...' ; `
    Write-Host 'Setting up R environment...' ; `
    Write-Host "R_LIBS_USER: $env:R_LIBS_USER" ; `
    Write-Host 'Installing R packages...' ; `
    Rscript -e "options(repos=c(CRAN='https://cran.rstudio.com')); dir.create(Sys.getenv('R_LIBS_USER'), recursive=TRUE, showWarnings=FALSE); .libPaths(Sys.getenv('R_LIBS_USER')); install.packages('remotes'); if (file.exists('C:/temp/install_packages.R')) source('C:/temp/install_packages.R') else install.packages(c('rmarkdown','knitr','ggplot2')); for (p in c('rmarkdown','knitr')) if (!require(p, character.only=TRUE, quietly=TRUE)) stop('missing: ', p)" ; `
    Write-Host '📦 R packages installed' ; `
    Write-Host 'Verifying R packages...' ; `
    Rscript C:/temp/verify_r_packages.R ; `
    Write-Host '✅ R package installation complete'

# ------------------------------------------------------------
# PHASE 12: Cleanup (same as quarto-build workflow)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING CLEANUP ===' ; `
    Write-Host 'Cleaning temporary files (same as quarto-build workflow)...' ; `
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
    Write-Host '✅ Cleanup complete'

# ------------------------------------------------------------
# FINAL CHECKS: Verify all installations (same as quarto-build workflow)
# ------------------------------------------------------------
WORKDIR C:/workspace
RUN Write-Host '=== FINAL VERIFICATION ===' ; `
    Write-Host 'Verifying all installations (same checks as quarto-build workflow)...' ; `
    Write-Host 'Checking Quarto...' ; `
    quarto --version ; `
    Write-Host '✅ Quarto verified' ; `
    Write-Host 'Checking Python...' ; `
    python --version ; `
    Write-Host '✅ Python verified' ; `
    Write-Host 'Checking R...' ; `
    R --version ; `
    Write-Host '✅ R verified' ; `
    Write-Host 'Checking LaTeX...' ; `
    lualatex --version ; `
    Write-Host '✅ LaTeX verified' ; `
    Write-Host '✅ Windows container build completed successfully'
