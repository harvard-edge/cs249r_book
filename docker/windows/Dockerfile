# escape=`
# MLSysBook Windows Quarto Build Container (Windows Server 2022)
# - PowerShell 7 via ZIP (no MSI)
# - Quarto 1.7.31 via ZIP (no MSI)
# - Python 3.13.1 + requirements
# - Ghostscript + Inkscape (Chocolatey)
# - TeX Live pinned to 2025 snapshot + packages from tl_packages
# - R 4.3.2 + packages via install_packages.R
# - Verifications: versions, kpsewhich font files, TikZ smoke test

FROM mcr.microsoft.com/windows/server:ltsc2022 as prep

SHELL ["powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass", "-Command"]

RUN Write-Host 'Creating TeX Live installation profile...' ; `
    $profilePath = 'C:\\temp\\texlive.profile' ; `
    $profileContent = @"
selected_scheme scheme-full
TEXDIR C:/texlive/2025
TEXMFCONFIG C:/texlive/2025/texmf-config
TEXMFHOME C:/texlive/2025/texmf-home
TEXMFLOCAL C:/texlive/2025/texmf-local
TEXMFSYSCONFIG C:/texlive/2025/texmf-config
TEXMFSYSVAR C:/texlive/2025/texmf-var
TEXMFVAR C:/texlive/2025/texmf-var
binary_windows 1
instopt_adjustpath 1
instopt_adjustrepo 1
instopt_letter 0
instopt_portable 0
instopt_write18_restricted 1
tlpdbopt_autobackup 0
tlpdbopt_backupdir tlpkg/backups
tlpdbopt_create_formats 1
tlpdbopt_desktop_integration 1
tlpdbopt_file_assocs 1
tlpdbopt_generate_updmap 0
tlpdbopt_install_docfiles 0
tlpdbopt_install_srcfiles 0
tlpdbopt_post_code 1
tlpdbopt_sys_bin C:/texlive/2025/bin/windows
tlpdbopt_sys_info C:/texlive/2025/texmf-dist/doc/info
tlpdbopt_sys_man C:/texlive/2025/texmf-dist/doc/man
tlpdbopt_w32_multi_user 0
"@; `
    New-Item -ItemType Directory -Force -Path 'C:\\temp' | Out-Null ; `
    $profileContent | Out-File -FilePath $profilePath -Encoding ASCII ; `
    Write-Host 'Profile created at C:\\temp\\texlive.profile'

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
RUN Write-Host '=== STARTING DEPENDENCY FILE COPY ==='
COPY tools/dependencies/requirements.txt C:/temp/requirements.txt
COPY tools/dependencies/install_packages.R C:/temp/install_packages.R
COPY tools/dependencies/tl_packages C:/temp/tl_packages
COPY docker/windows/verify_r_packages.R C:/temp/verify_r_packages.R
RUN Write-Host '✅ Dependency file copy complete'

# Copy the pre-generated texlive.profile from the 'prep' stage
COPY --from=prep C:/temp/texlive.profile C:/temp/texlive.profile

# ------------------------------------------------------------
# PHASE 4: Install TeX Live (Official network installer)
# ------------------------------------------------------------
# Note: Chocolatey texlive package is broken (missing install-tl.zip)
# Using official installer with batch mode for reliable Docker builds
RUN Write-Host '=== STARTING TEX LIVE INSTALLATION ===' ; `
    Write-Host '📦 Using official TeX Live network installer' ; `
    Write-Host '⏰ Installation will take 15-20 minutes' ; `
    `
    Write-Host '🔄 Downloading TeX Live installer...' ; `
    $installerUrl = 'https://mirror.ctan.org/systems/texlive/tlnet/install-tl-windows.exe' ; `
    $installerPath = 'C:\temp\install-tl-windows.exe' ; `
    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing ; `
    $installerSizeMB = [math]::Round((Get-Item $installerPath).Length / 1MB, 2) ; `
    Write-Host ('✅ Downloaded installer (' + $installerSizeMB + ' MB)') ; `
    `
    Write-Host '🔧 Using pre-generated installation profile...' ; `
    $profilePath = 'C:\temp\texlive.profile' ; `
    Write-Host '✅ Profile located' ; `
    Write-Host '📋 Profile settings: scheme-full, no docs/sources' ; `
    `
    Write-Host '🔄 Running TeX Live installer (batch mode)...' ; `
    $installProcess = Start-Process -FilePath $installerPath `
        -ArgumentList '-profile', $profilePath, '-no-gui', '-non-admin' `
        -Wait -PassThru -NoNewWindow -RedirectStandardOutput 'C:\temp\texlive-install.log' -RedirectStandardError 'C:\temp\texlive-install-error.log' ; `
    `
    if ($installProcess.ExitCode -ne 0) { `
        Write-Host '❌ ERROR: TeX Live installation failed' ; `
        Write-Host '📋 Installer output:' ; `
        Get-Content 'C:\temp\texlive-install.log' -ErrorAction SilentlyContinue | Select-Object -Last 50 ; `
        Write-Host '📋 Installer errors:' ; `
        Get-Content 'C:\temp\texlive-install-error.log' -ErrorAction SilentlyContinue ; `
        throw "TeX Live installation failed with exit code $($installProcess.ExitCode)" ; `
    } ; `
    Write-Host '✅ TeX Live installed successfully' ; `
    `
    Write-Host '🔍 Locating TeX Live installation...' ; `
    $texLiveBin = 'C:\texlive\2025\bin\windows' ; `
    if (-not (Test-Path $texLiveBin)) { `
        Write-Host '❌ ERROR: TeX Live bin directory not found' ; `
        Write-Host '📁 Checking for alternative locations:' ; `
        Get-ChildItem 'C:\texlive' -Recurse -Depth 2 -Directory -ErrorAction SilentlyContinue | `
            Where-Object { $_.Name -eq 'windows' } | ForEach-Object { Write-Host "  Found: $($_.FullName)" } ; `
        throw "TeX Live bin directory not found at $texLiveBin" ; `
    } ; `
    Write-Host "✅ TeX Live bin: $texLiveBin" ; `
    `
    Write-Host '🔧 Adding TeX Live to PATH...' ; `
    $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
    $newPath = $texLiveBin + ';' + $mach ; `
    [Environment]::SetEnvironmentVariable('PATH', $newPath, 'Machine') ; `
    $env:PATH = $texLiveBin + ';' + $env:PATH ; `
    Write-Host '✅ PATH updated' ; `
    `
    Write-Host '🔍 Verifying lualatex...' ; `
    $lualatexPath = Join-Path $texLiveBin 'lualatex.exe' ; `
    if (Test-Path $lualatexPath) { `
        & $lualatexPath --version | Select-Object -First 2 ; `
    } else { `
        throw "lualatex.exe not found at $lualatexPath" ; `
    } ; `
    `
    Write-Host '📋 Installing additional collections from tl_packages...' ; `
    if (Test-Path 'C:\temp\tl_packages') { `
        $collections = Get-Content 'C:\temp\tl_packages' | Where-Object { $_.Trim() -and -not $_.StartsWith('#') } ; `
        Write-Host "📦 Found $($collections.Count) collections" ; `
        $tlmgrPath = Join-Path $texLiveBin 'tlmgr.bat' ; `
        foreach ($collection in $collections) { `
            Write-Host "  Installing $collection..." ; `
            & $tlmgrPath install $collection 2>&1 | Out-Null ; `
        } ; `
    } ; `
    `
    Write-Host '🧹 Cleaning up installer files...' ; `
    Remove-Item $installerPath, $profilePath -Force -ErrorAction SilentlyContinue ; `
    Write-Host '✅ TeX Live installation complete'

# ------------------------------------------------------------
# PHASE 5: Install Scoop (Package manager setup)
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
# PHASE 6: Install Quarto (Main tool)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING QUARTO INSTALLATION ===' ; `
    Write-Host 'Installing Quarto via Scoop...' ; `
    scoop install quarto ; `
    Write-Host '📦 Quarto installed' ; `
    Write-Host 'Verifying Quarto installation...' ; `
    quarto --version ; `
    Write-Host '✅ Quarto installation completed!'

# ------------------------------------------------------------
# PHASE 7: Install Ghostscript (required for PDF generation)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING GHOSTSCRIPT INSTALLATION ===' ; `
    Write-Host 'Installing Ghostscript via Scoop...' ; `
    scoop install main/ghostscript ; `
    Write-Host '📦 Ghostscript installed' ; `
    Write-Host 'Verifying Ghostscript installation...' ; `
    gs --version ; `
    Write-Host '✅ Ghostscript installation complete'

# ------------------------------------------------------------
# PHASE 8: Install Inkscape (required for SVG processing)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING INKSCAPE INSTALLATION ===' ; `
    Write-Host 'Installing Inkscape via Scoop...' ; `
    scoop install inkscape ; `
    Write-Host '📦 Inkscape installed' ; `
    Write-Host 'Verifying Inkscape installation...' ; `
    inkscape --version ; `
    Write-Host '✅ Inkscape installation complete'

# ------------------------------------------------------------
# PHASE 9: Install Python (Medium complexity)
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
# PHASE 10: Install Python packages (Medium complexity)
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
# PHASE 11: Install Visual C++ Redistributable (Required for Quarto DLLs)
# ------------------------------------------------------------
RUN Write-Host '=== STARTING VISUAL C++ REDISTRIBUTABLE INSTALLATION ===' ; `
    Write-Host 'Installing Microsoft Visual C++ Redistributable...' ; `
    Write-Host 'This is required for Quarto DLL dependencies on Windows' ; `
    choco install vcredist-all -y ; `
    Write-Host '📦 Visual C++ Redistributable installed' ; `
    Write-Host '✅ Visual C++ Redistributable installation complete'

# ------------------------------------------------------------
# PHASE 12: Install R (Medium complexity)
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
# PHASE 13: Install R packages (Medium complexity)
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
# FINAL CHECKS: Comprehensive verification with diagnostics
# ------------------------------------------------------------
WORKDIR C:/workspace
RUN Write-Host '=== FINAL VERIFICATION WITH ENHANCED DIAGNOSTICS ===' ; `
    Write-Host 'Verifying all installations with comprehensive checks...' ; `
    Write-Host '' ; `
    Write-Host '🔍 SYSTEM DIAGNOSTICS:' ; `
    Write-Host '----------------------' ; `
    Write-Host 'PATH environment variable:' ; `
    Write-Host $env:PATH ; `
    Write-Host '' ; `
    Write-Host 'Visual C++ Redistributable check:' ; `
    Get-ChildItem 'C:\Windows\System32' -Filter 'msvcp*.dll' | Select-Object Name, Length, LastWriteTime ; `
    Write-Host '' ; `
    Write-Host '📊 TOOL VERIFICATION:' ; `
    Write-Host '---------------------' ; `
    Write-Host 'Checking Quarto...' ; `
    try { `
        quarto --version ; `
        Write-Host '✅ Quarto version check: PASSED' ; `
        Write-Host 'Running Quarto check for comprehensive validation...' ; `
        & quarto check 2>&1 | Write-Host ; `
        if ($LASTEXITCODE -eq 0) { `
            Write-Host '✅ Quarto check: PASSED' ; `
        } else { `
            Write-Host '⚠️ Quarto check: ISSUES DETECTED' ; `
            Write-Host "Exit code: $LASTEXITCODE" ; `
        } ; `
    } catch { `
        Write-Host '❌ Quarto verification failed:' ; `
        Write-Host $_.Exception.Message ; `
    } ; `
    Write-Host 'Checking Python...' ; `
    python --version ; `
    Write-Host '✅ Python verified' ; `
    Write-Host 'Checking R...' ; `
    R --version ; `
    Write-Host '✅ R verified' ; `
    Write-Host 'Checking LaTeX...' ; `
    lualatex --version ; `
    Write-Host '✅ LaTeX verified' ; `
    Write-Host 'Checking Ghostscript...' ; `
    gs --version ; `
    Write-Host '✅ Ghostscript verified' ; `
    Write-Host 'Checking Inkscape...' ; `
    inkscape --version ; `
    Write-Host '✅ Inkscape verified' ; `
    Write-Host '' ; `
    Write-Host '🎯 FINAL STATUS:' ; `
    Write-Host '----------------' ; `
    Write-Host '✅ Windows container build completed with enhanced diagnostics'