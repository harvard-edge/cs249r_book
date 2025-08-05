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
ENV R_LIBS_USER=C:/r-lib
ENV QUARTO_LOG_LEVEL=INFO
ENV PYTHONIOENCODING=utf-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Copy dependency files needed for installations
COPY tools/dependencies/requirements/ C:/temp/requirements/
COPY tools/dependencies/requirements-build.txt C:/temp/requirements.txt
COPY tools/dependencies/install_packages.R C:/temp/install_packages.R  
COPY tools/dependencies/tl_packages C:/temp/tl_packages
COPY docker/quarto-build-windows/verify_r_packages.R C:/temp/verify_r_packages.R

# Create R library directory
RUN pwsh -Command "New-Item -ItemType Directory -Path 'C:/r-lib' -Force"

# Setup Quarto (using exact version from workflow)
RUN pwsh -Command "Invoke-WebRequest -Uri 'https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.31/quarto-1.7.31-win.zip' -OutFile 'quarto.zip'" && \
    pwsh -Command "Expand-Archive -Path 'quarto.zip' -DestinationPath 'C:\quarto' -Force" && \
    pwsh -Command "del quarto.zip" && \
    setx PATH "%PATH%;C:\quarto\quarto-1.7.31\bin" /M

# Setup Python 3.13 (same as workflow with progress indicators)
RUN pwsh -Command " \
    Write-Host '🚀 === STARTING PYTHON 3.13 INSTALLATION ==='; \
    Write-Host '⏰ Estimated time: 2-3 minutes'; \
    $startTime = Get-Date; \
    Write-Host '📦 Installing Python 3.13.1 (exact version from workflow)...'; \
    " && \
    choco install -y python --version=3.13.1 && \
    pwsh -Command " \
    # Refresh environment variables to pick up Python PATH \
    $env:PATH = [System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::Machine) + ';' + [System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::User); \
    # Add common Python paths explicitly \
    $env:PATH += ';C:\Python313;C:\Python313\Scripts;C:\ProgramData\chocolatey\bin'; \
    \
    $endTime = Get-Date; \
    $duration = ($endTime - $startTime).TotalMinutes; \
    Write-Host ('✅ === PYTHON 3.13 INSTALLATION COMPLETE === (' + $duration.ToString('F1') + ' minutes)'); \
    \
    # Test Python installation with explicit path handling \
    try { \
        $pythonVersion = & python --version 2>&1; \
        Write-Host ('📊 Python version: ' + $pythonVersion); \
        $pipVersion = & python -m pip --version 2>&1; \
        Write-Host ('📊 Pip version: ' + $pipVersion); \
    } catch { \
        Write-Host '⚠️ Python installed but not immediately available in PATH (normal for containers)'; \
        Write-Host 'Python will be available in subsequent RUN commands'; \
    } \
    "

# Install Python dependencies (same as workflow with progress indicators)
RUN pwsh -Command " \
    Write-Host '🚀 === STARTING PYTHON PACKAGE INSTALLATION ==='; \
    Write-Host '⏰ Estimated time: 1-2 minutes'; \
    $startTime = Get-Date; \
    \
    # Ensure Python is in PATH \
    $env:PATH = [System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::Machine) + ';C:\Python313;C:\Python313\Scripts;C:\ProgramData\chocolatey\bin'; \
    \
    Write-Host '🔄 Upgrading pip...'; \
    & python -m pip install --upgrade pip; \
    \
    Write-Host '📊 Analyzing requirements.txt...'; \
    $packageCount = (Get-Content 'C:/temp/requirements.txt' | Where-Object { $_ -notmatch '^#' -and $_ -ne '' }).Count; \
    Write-Host ('📦 Found ' + $packageCount + ' Python packages to install'); \
    \
    Write-Host '🔄 Installing Python packages...'; \
    & python -m pip install -r C:/temp/requirements.txt; \
    \
    $endTime = Get-Date; \
    $duration = ($endTime - $startTime).TotalMinutes; \
    Write-Host ('✅ === PYTHON PACKAGES COMPLETE === (' + $duration.ToString('F1') + ' minutes)'); \
    \
    try { \
        $packageCount = (& python -m pip list | Measure-Object).Count; \
        Write-Host ('📊 Total packages: ' + $packageCount); \
    } catch { \
        Write-Host '📊 Python packages installed successfully'; \
    } \
    "

# Install Ghostscript (direct download - most reliable for containers)
RUN pwsh -Command " \
    Write-Host '🚀 === STARTING GHOSTSCRIPT INSTALLATION ==='; \
    Write-Host '⏰ Estimated time: 1-2 minutes (direct download)'; \
    Write-Host 'Free disk space: ' + ((Get-WmiObject -Class Win32_LogicalDisk -Filter \"DeviceID='C:'\").FreeSpace / 1GB).ToString('F2') + ' GB'; \
    $startTime = Get-Date; \
    \
    Write-Host '📦 Downloading Ghostscript directly (bypassing package managers)...'; \
    $url = 'https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs10051/gs10051w64.exe'; \
    $installer = 'C:/temp/ghostscript.exe'; \
    \
    Write-Host '🔄 Starting download from GitHub releases...'; \
    try { \
        Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing; \
        $fileSize = (Get-Item $installer).Length / 1MB; \
        Write-Host ('📊 Downloaded: ' + $fileSize.ToString('F1') + ' MB'); \
    } catch { \
        Write-Host '⚠️ GitHub download failed, trying alternative source...'; \
        $altUrl = 'https://www.ghostscript.com/releases/gpcldnld.html'; \
        Write-Host '🔄 Using direct Ghostscript.com download...'; \
        $altUrl = 'https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs10051/gs10051w32.exe'; \
        Invoke-WebRequest -Uri $altUrl -OutFile $installer -UseBasicParsing; \
    } \
    \
    Write-Host '📦 Installing Ghostscript silently...'; \
    Write-Host 'Running: ' + $installer + ' /S /D=C:/Program Files/gs/gs10.05.1'; \
    Start-Process -FilePath $installer -ArgumentList '/S', '/D=C:/Program Files/gs/gs10.05.1' -Wait -NoNewWindow; \
    \
    Write-Host '🧹 Cleaning up installer...'; \
    Remove-Item $installer -ErrorAction SilentlyContinue; \
    \
    Write-Host '🔧 Configuring Ghostscript PATH...'; \
    $gsPath = Get-ChildItem 'C:/Program Files/gs' -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1; \
    if ($gsPath) { \
        $binPath = Join-Path $gsPath.FullName 'bin'; \
        Write-Host ('📍 Found Ghostscript at: ' + $binPath); \
        [Environment]::SetEnvironmentVariable('PATH', \"$env:PATH;$binPath\", [EnvironmentVariableTarget]::Machine); \
        \
        Write-Host '🧪 Testing Ghostscript installation...'; \
        try { \
            $gsVersion = & \"$binPath/gs.exe\" --version 2>&1; \
            Write-Host ('✅ Ghostscript version: ' + $gsVersion); \
        } catch { \
            Write-Host '⚠️ Ghostscript installed but test failed (normal in some containers)'; \
        } \
    } else { \
        Write-Host '⚠️ Ghostscript installation directory not found, checking alternative locations...'; \
        $altPath = 'C:/Program Files (x86)/gs'; \
        if (Test-Path $altPath) { \
            $gsPath = Get-ChildItem $altPath | Sort-Object Name -Descending | Select-Object -First 1; \
            $binPath = Join-Path $gsPath.FullName 'bin'; \
            Write-Host ('📍 Found Ghostscript at: ' + $binPath); \
            [Environment]::SetEnvironmentVariable('PATH', \"$env:PATH;$binPath\", [EnvironmentVariableTarget]::Machine); \
        } else { \
            Write-Host '❌ Ghostscript installation failed - directory not found'; \
        } \
    } \
    \
    $endTime = Get-Date; \
    $duration = ($endTime - $startTime).TotalMinutes; \
    Write-Host ('✅ === GHOSTSCRIPT INSTALLATION COMPLETE === (' + $duration.ToString('F1') + ' minutes)'); \
    "

# Install Inkscape (using exact method from workflow with progress indicators)
RUN pwsh -Command " \
    Write-Host '🚀 === STARTING INKSCAPE INSTALLATION ==='; \
    Write-Host '⏰ Estimated time: 2-3 minutes'; \
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; \
    $OutputEncoding = [System.Text.Encoding]::UTF8; \
    [Console]::InputEncoding = [System.Text.Encoding]::UTF8; \
    $startTime = Get-Date; \
    \
    try { \
        Write-Host '🔄 Attempting winget installation (may be slow)...'; \
        Write-Host '📦 Downloading Inkscape package...'; \
        winget install Inkscape.Inkscape --accept-source-agreements --accept-package-agreements; \
        Write-Host '✅ Inkscape installed via winget'; \
    } catch { \
        Write-Host '🔄 winget failed or unavailable, using chocolatey...'; \
        Write-Host '📦 Installing via chocolatey (alternative method)...'; \
        choco install inkscape -y; \
        Write-Host '✅ Inkscape installed via chocolatey'; \
    } \
    \
    Write-Host '🔧 Adding Inkscape to PATH...'; \
    [Environment]::SetEnvironmentVariable('PATH', \"$env:PATH;C:\Program Files\Inkscape\bin\", [EnvironmentVariableTarget]::Machine); \
    \
    $endTime = Get-Date; \
    $duration = ($endTime - $startTime).TotalMinutes; \
    Write-Host ('✅ === INKSCAPE INSTALLATION COMPLETE === (' + $duration.ToString('F1') + ' minutes)'); \
    "

# Install TeX Live packages (using zauguin/install-texlive equivalent with progress indicators)
RUN pwsh -Command " \
    Write-Host '🚀 === STARTING TEX LIVE INSTALLATION ==='; \
    Write-Host '⏰ Estimated time: 8-12 minutes (largest phase)'; \
    Write-Host 'Free disk space before: ' + ((Get-WmiObject -Class Win32_LogicalDisk -Filter \"DeviceID='C:'\").FreeSpace / 1GB).ToString('F2') + ' GB'; \
    $startTime = Get-Date; \
    \
    Write-Host '🔄 Downloading TeX Live installer...'; \
    Invoke-WebRequest -Uri 'https://mirror.ctan.org/systems/texlive/tlnet/install-tl-windows.exe' -OutFile 'C:/temp/install-tl-windows.exe'; \
    \
    Write-Host '🔧 Creating TeX Live installation profile...'; \
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
'@ | Out-File -FilePath 'C:/temp/texlive.profile' -Encoding ASCII; \
    \
    Write-Host '📦 Installing TeX Live base system (this may take 3-5 minutes)...'; \
    Start-Process -FilePath 'C:/temp/install-tl-windows.exe' -ArgumentList '-profile', 'C:/temp/texlive.profile' -Wait; \
    \
    Write-Host '🔧 Adding TeX Live to PATH...'; \
    [Environment]::SetEnvironmentVariable('PATH', \"$env:PATH;C:\texlive\bin\windows\", [EnvironmentVariableTarget]::Machine); \
    \
    Write-Host '📊 Analyzing tl_packages file...'; \
    $packages = Get-Content 'C:/temp/tl_packages' | Where-Object { $_.Trim() -ne '' }; \
    $packageCount = $packages.Count; \
    Write-Host ('📦 Found ' + $packageCount + ' TeX Live collections to install'); \
    \
    Write-Host '🔄 Installing TeX Live collections...'; \
    $i = 1; \
    foreach ($package in $packages) { \
        Write-Host ('📦 [' + $i + '/' + $packageCount + '] Installing: ' + $package.Trim()); \
        & 'C:/texlive/bin/windows/tlmgr.bat' install $package.Trim(); \
        $i++; \
    }; \
    \
    Write-Host '🧹 Cleaning up TeX Live installer...'; \
    Remove-Item 'C:/temp/install-tl-windows.exe'; \
    Remove-Item 'C:/temp/texlive.profile'; \
    \
    $endTime = Get-Date; \
    $duration = ($endTime - $startTime).TotalMinutes; \
    Write-Host ('✅ === TEX LIVE INSTALLATION COMPLETE === (' + $duration.ToString('F1') + ' minutes)'); \
    Write-Host 'Free disk space after: ' + ((Get-WmiObject -Class Win32_LogicalDisk -Filter \"DeviceID='C:'\").FreeSpace / 1GB).ToString('F2') + ' GB'; \
    "

# Setup R (using exact version from workflow with progress indicators)
RUN pwsh -Command " \
    Write-Host '🚀 === STARTING R INSTALLATION ==='; \
    Write-Host '⏰ Estimated time: 2-3 minutes'; \
    $startTime = Get-Date; \
    Write-Host '📦 Installing R 4.4.1 (exact version from workflow)...'; \
    " && \
    choco install -y r --version=4.4.1 && \
    pwsh -Command " \
    $endTime = Get-Date; \
    $duration = ($endTime - $startTime).TotalMinutes; \
    Write-Host ('✅ === R INSTALLATION COMPLETE === (' + $duration.ToString('F1') + ' minutes)'); \
    Write-Host ('📊 R version: ' + (R --version | Select-Object -First 1)); \
    "

# Install R packages (using simplified workflow approach with progress indicators)
RUN pwsh -Command " \
    Write-Host '🚀 === STARTING R PACKAGE INSTALLATION ==='; \
    Write-Host '⏰ Estimated time: 3-5 minutes'; \
    $startTime = Get-Date; \
    " && \
    Rscript -e " \
    options(repos = c(CRAN = 'https://cran.rstudio.com')); \
    cat('📦 Starting R package installation...\n'); \
    cat(paste('📍 R library path:', Sys.getenv('R_LIBS_USER'), '\n')); \
    \
    lib_path <- Sys.getenv('R_LIBS_USER'); \
    dir.create(lib_path, showWarnings = FALSE, recursive = TRUE); \
    .libPaths(lib_path); \
    \
    cat('📦 [1/2] Installing remotes package...\n'); \
    install.packages('remotes'); \
    \
    if (file.exists('C:/temp/install_packages.R')) { \
      cat('📦 [2/2] Installing packages from tools/dependencies/install_packages.R...\n'); \
      source('C:/temp/install_packages.R'); \
    } else { \
      cat('⚠️ No install_packages.R found, installing basic packages\n'); \
      install.packages(c('rmarkdown', 'knitr', 'ggplot2')); \
    }; \
    \
    cat('✅ R package installation complete\n'); \
    ip <- installed.packages()[, 'Package']; \
    cat(paste('📊 Total R packages installed:', length(ip), '\n')); \
    " && \
    pwsh -Command " \
    $endTime = Get-Date; \
    $duration = ($endTime - $startTime).TotalMinutes; \
    Write-Host ('✅ === R PACKAGE INSTALLATION COMPLETE === (' + $duration.ToString('F1') + ' minutes)'); \
    "

# Verify R package installation
RUN pwsh -Command "Write-Host '🔍 Verifying R package installation...'" && \
    Rscript C:/temp/verify_r_packages.R

# Clean up
RUN del C:/temp/requirements.txt C:/temp/install_packages.R C:/temp/verify_r_packages.R C:/temp/tl_packages

# Set working directory
WORKDIR C:/workspace

# Final verification (using same tools that workflow uses)
RUN quarto --version && \
    python --version && \
    R --version && \
    lualatex --version

# Health check
RUN pwsh -Command "Write-Host '✅ Windows container build completed successfully'"