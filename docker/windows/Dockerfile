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
RUN Write-Host '=== STARTING DEPENDENCY FILE COPY ==='
COPY tools/dependencies/requirements.txt C:/temp/requirements.txt
COPY tools/dependencies/install_packages.R C:/temp/install_packages.R
COPY tools/dependencies/tl_packages C:/temp/tl_packages
COPY docker/windows/verify_r_packages.R C:/temp/verify_r_packages.R
RUN Write-Host '✅ Dependency file copy complete'

# ------------------------------------------------------------
# PHASE 4: Install TeX Live via Chocolatey with full scheme
# ------------------------------------------------------------
RUN Write-Host '=== STARTING TEX LIVE INSTALLATION ===' ; `
    Write-Host '📦 Installing TeX Live via Chocolatey with FULL scheme...' ; `
    Write-Host '⏰ This will take 15-20 minutes for full installation' ; `
    choco install texlive --params "'/scheme:full'" -y ; `
    Write-Host '✅ TeX Live (full scheme) installed via Chocolatey' ; `
    `
    Write-Host '🔍 Locating TeX Live installation...' ; `
    $texLiveBin = Get-ChildItem -Path 'C:\texlive' -Recurse -Depth 3 -Directory -ErrorAction SilentlyContinue | `
        Where-Object { $_.Name -eq 'windows' -and $_.FullName -like '*\bin\windows' } | `
        Select-Object -First 1 -ExpandProperty FullName ; `
    if (-not $texLiveBin) { `
        Write-Host '❌ ERROR: TeX Live bin directory not found' ; `
        Write-Host '📁 Searching C:\texlive for bin directories:' ; `
        Get-ChildItem 'C:\texlive' -Recurse -Depth 3 -Directory -ErrorAction SilentlyContinue | `
            Where-Object { $_.Name -eq 'windows' } | ForEach-Object { Write-Host "  Found: $($_.FullName)" } ; `
        throw 'TeX Live bin directory not found' ; `
    } ; `
    Write-Host "✅ Found TeX Live bin: $texLiveBin" ; `
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