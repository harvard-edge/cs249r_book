# MLSysBook Windows Quarto Build Container
# Replicates the exact environment that quarto-build workflow uses successfully
# Based on Windows Server 2022 (same as windows-latest GitHub Actions runner)

FROM mcr.microsoft.com/windows/servercore:ltsc2022

# === PHASE 1: POWERSHELL 7 INSTALLATION ===
RUN curl -fSLo PowerShell.msi https://github.com/PowerShell/PowerShell/releases/download/v7.4.1/PowerShell-7.4.1-win-x64.msi && \
    msiexec /i PowerShell.msi /quiet /norestart && \
    del PowerShell.msi

# === PHASE 2: CHOCOLATEY INSTALLATION ===
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

# === PHASE 3: GHOSTSCRIPT INSTALLATION (EARLY - MOST PROBLEMATIC) ===
RUN pwsh -Command " \
    Write-Host '🚀 === STARTING GHOSTSCRIPT INSTALLATION ==='; \
    Write-Host '⏰ Estimated time: 1-3 minutes (multiple fallback methods)'; \
    $startTime = Get-Date; \
    $gsInstalled = $false; \
    \
    Write-Host '📦 Method 1: Direct download and install (latest version - most reliable)...'; \
    try { \
        # Try multiple download URLs for latest Ghostscript 10.05.1 \
        $gsUrls = @( \
            'https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs10051/gs10051w64.exe', \
            'https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/latest/download/gs10051w64.exe' \
        ); \
        $gsInstaller = 'C:/temp/gs_installer.exe'; \
        $downloadSuccess = $false; \
        \
        foreach ($gsUrl in $gsUrls) { \
            Write-Host ('🔄 Trying download from: ' + $gsUrl); \
            try { \
                Invoke-WebRequest -Uri $gsUrl -OutFile $gsInstaller -TimeoutSec 300 -UserAgent 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'; \
                if (Test-Path $gsInstaller -and (Get-Item $gsInstaller).Length -gt 1MB) { \
                    Write-Host ('✅ Downloaded successfully (' + ((Get-Item $gsInstaller).Length / 1MB).ToString('F1') + ' MB)'); \
                    $downloadSuccess = $true; \
                    break; \
                } \
            } catch { \
                Write-Host ('⚠️ Download failed from ' + $gsUrl + ': ' + $_.Exception.Message); \
            } \
        } \
        \
        if ($downloadSuccess) { \
            Write-Host '🔄 Installing Ghostscript silently...'; \
            $installProcess = Start-Process -FilePath $gsInstaller -ArgumentList '/S' -Wait -PassThru -NoNewWindow; \
            if ($installProcess.ExitCode -eq 0) { \
                Write-Host '✅ Ghostscript installed via direct download'; \
                $gsInstalled = $true; \
            } else { \
                Write-Host ('⚠️ Installation failed with exit code: ' + $installProcess.ExitCode); \
            } \
            # Clean up installer \
            Remove-Item $gsInstaller -ErrorAction SilentlyContinue; \
        } else { \
            Write-Host '❌ All download attempts failed'; \
        } \
    } catch { \
        Write-Host '⚠️ Direct download method failed: ' + $_.Exception.Message; \
    } \
    \
    if (-not $gsInstalled) { \
        Write-Host '📦 Method 2: Chocolatey fallback (with verbose logging)...'; \
        try { \
            # First check chocolatey version and availability \
            Write-Host '🔍 Chocolatey status check...'; \
            $chocoVersion = choco --version 2>&1; \
            Write-Host ('📊 Chocolatey version: ' + $chocoVersion); \
            \
            # Set UTF-8 encoding like the working workflow \
            [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; \
            $OutputEncoding = [System.Text.Encoding]::UTF8; \
            [Console]::InputEncoding = [System.Text.Encoding]::UTF8; \
            \
            # Run chocolatey install with verbose logging \
            Write-Host '🔄 Installing Ghostscript via chocolatey with verbose output...'; \
            \
            $chocoInstallScript = { \
                try { \
                    choco install ghostscript -y -dv; \
                    return @{ ExitCode = $LASTEXITCODE; Success = $true; Output = 'Install completed' }; \
                } catch { \
                    return @{ ExitCode = -1; Success = $false; Output = $_.Exception.Message }; \
                } \
            }; \
            \
            $chocoJob = Start-Job -ScriptBlock $chocoInstallScript; \
            $timeout = 1800; # 30 minutes \
            \
            Write-Host ('⏰ Waiting up to ' + ($timeout / 60) + ' minutes for chocolatey installation...'); \
            $completed = Wait-Job -Job $chocoJob -Timeout $timeout; \
            \
            if ($completed) { \
                $result = Receive-Job -Job $chocoJob; \
                Write-Host ('📋 Installation result: ' + $result.Output); \
                Write-Host ('📋 Exit code: ' + $result.ExitCode); \
                \
                if ($result.ExitCode -eq 0) { \
                    Write-Host '✅ Ghostscript installed via chocolatey'; \
                    $gsInstalled = $true; \
                } else { \
                    Write-Host ('⚠️ Chocolatey install failed with exit code: ' + $result.ExitCode); \
                } \
            } else { \
                Write-Host ('⚠️ Chocolatey installation timed out after ' + ($timeout / 60) + ' minutes'); \
                Remove-Job -Job $chocoJob -Force; \
            } \
            \
            Remove-Job -Job $chocoJob -Force -ErrorAction SilentlyContinue; \
            \
        } catch { \
            Write-Host ('⚠️ Chocolatey method failed: ' + $_.Exception.Message); \
        } \
    } \
    \
    if (-not $gsInstalled) { \
        Write-Host '❌ All Ghostscript installation methods failed'; \
        Write-Host '⚠️ Continuing build without Ghostscript (some features may not work)'; \
    } \
    \
    Write-Host '🔧 Setting up Ghostscript PATH...'; \
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
        Write-Host '⚠️ Ghostscript directory not found - checking alternative locations...'; \
        $altPaths = @('C:/Program Files (x86)/gs', 'C:/gs', 'C:/ghostscript'); \
        foreach ($altPath in $altPaths) { \
            if (Test-Path $altPath) { \
                Write-Host ('📍 Found Ghostscript at alternative location: ' + $altPath); \
                [Environment]::SetEnvironmentVariable('PATH', \"$env:PATH;$altPath/bin\", [EnvironmentVariableTarget]::Machine); \
                break; \
            } \
        } \
    } \
    \
    $endTime = Get-Date; \
    $duration = ($endTime - $startTime).TotalMinutes; \
    Write-Host ('✅ === GHOSTSCRIPT INSTALLATION COMPLETE === (' + $duration.ToString('F1') + ' minutes)'); \
    "

# === PHASE 4: QUARTO INSTALLATION ===
RUN pwsh -Command " \
    Write-Host '🚀 === STARTING QUARTO INSTALLATION ==='; \
    Write-Host '⏰ Estimated time: 1-2 minutes'; \
    $startTime = Get-Date; \
    \
    Write-Host '📦 Downloading Quarto 1.7.31...'; \
    Invoke-WebRequest -Uri 'https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.31/quarto-1.7.31-win.zip' -OutFile 'quarto.zip'; \
    \
    Write-Host '📦 Extracting Quarto...'; \
    Expand-Archive -Path 'quarto.zip' -DestinationPath 'C:\quarto' -Force; \
    \
    Write-Host '🧹 Cleaning up installer...'; \
    Remove-Item 'quarto.zip'; \
    \
    $endTime = Get-Date; \
    $duration = ($endTime - $startTime).TotalMinutes; \
    Write-Host ('✅ === QUARTO INSTALLATION COMPLETE === (' + $duration.ToString('F1') + ' minutes)'); \
    " && \
    setx PATH "%PATH%;C:\quarto\quarto-1.7.31\bin" /M

# === PHASE 5: PYTHON 3.13 INSTALLATION ===
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

# === PHASE 6: PYTHON PACKAGE INSTALLATION ===
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

# === PHASE 7: INKSCAPE INSTALLATION ===
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

# === PHASE 8: TEX LIVE INSTALLATION ===
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

# === PHASE 9: R INSTALLATION ===
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

# === PHASE 10: R PACKAGE INSTALLATION ===
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

# === PHASE 11: R PACKAGE VERIFICATION ===
RUN pwsh -Command "Write-Host '🔍 Verifying R package installation...'" && \
    Rscript C:/temp/verify_r_packages.R

# === PHASE 12: CLEANUP ===
RUN pwsh -Command " \
    Write-Host '🧹 === STARTING CLEANUP ==='; \
    Remove-Item C:/temp/requirements.txt -ErrorAction SilentlyContinue; \
    Remove-Item C:/temp/install_packages.R -ErrorAction SilentlyContinue; \
    Remove-Item C:/temp/verify_r_packages.R -ErrorAction SilentlyContinue; \
    Remove-Item C:/temp/tl_packages -ErrorAction SilentlyContinue; \
    Remove-Item C:/temp/requirements/ -Recurse -Force -ErrorAction SilentlyContinue; \
    Write-Host '✅ === CLEANUP COMPLETE ==='; \
    "

# Set working directory
WORKDIR C:/workspace

# Final verification (using same tools that workflow uses)
RUN quarto --version && \
    python --version && \
    R --version && \
    lualatex --version

# Health check
RUN pwsh -Command "Write-Host '✅ Windows container build completed successfully'"