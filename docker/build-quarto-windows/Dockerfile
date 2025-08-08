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
# PHASE 0: Base dirs and env
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
RUN Write-Host 'Installing PowerShell 7 ZIP...' ; `
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
# PHASE 2: Chocolatey
# ------------------------------------------------------------
RUN Write-Host 'Installing Chocolatey...' ; `
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 ; `
    iex ((New-Object Net.WebClient).DownloadString('https://chocolatey.org/install.ps1')) ; `
    choco --version

# ------------------------------------------------------------
# PHASE 3: Copy dependency files
# ------------------------------------------------------------
COPY tools/dependencies/requirements/                  C:/temp/requirements/
COPY tools/dependencies/requirements-build.txt         C:/temp/requirements.txt
COPY tools/dependencies/install_packages.R             C:/temp/install_packages.R
COPY tools/dependencies/tl_packages                    C:/temp/tl_packages
COPY docker/build-quarto-windows/verify_r_packages.R   C:/temp/verify_r_packages.R

# ------------------------------------------------------------
# PHASE 4: Quarto 1.7.31 (ZIP, robust PATH)
# ------------------------------------------------------------
RUN Write-Host 'Installing Quarto 1.7.31...' ; `
    Invoke-WebRequest -Uri 'https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.31/quarto-1.7.31-win.zip' -OutFile 'C:\quarto.zip' -TimeoutSec 900 ; `
    Expand-Archive -Path 'C:\quarto.zip' -DestinationPath 'C:\quarto' -Force ; `
    Remove-Item 'C:\quarto.zip' -Force ; `
    Write-Host 'Searching for Quarto installation...' ; `
    $quartoFound = $false ; `
    $quartoPaths = @() ; `
    # Check our extracted location first
    $extractedDirs = Get-ChildItem -Directory C:\quarto -ErrorAction SilentlyContinue ; `
    foreach ($dir in $extractedDirs) { $quartoPaths += Join-Path $dir.FullName 'bin' } ; `
    # Check common Windows installation locations
    $quartoPaths += 'C:\Program Files\Quarto\bin' ; `
    $quartoPaths += 'C:\Program Files (x86)\Quarto\bin' ; `
    $quartoPaths += 'C:\Users\ContainerAdministrator\AppData\Local\Programs\Quarto\bin' ; `
    $quartoPaths += 'C:\Users\ContainerAdministrator\AppData\Roaming\Quarto\bin' ; `
    # Check if quarto.exe exists in any of these paths
    Write-Host 'Checking for Quarto installation in the following locations:' ; `
    foreach ($path in $quartoPaths) { Write-Host "  - $path" } ; `
    $foundPath = $null ; `
    foreach ($path in $quartoPaths) { `
        $quartoExe = Join-Path $path 'quarto.exe' ; `
        if (Test-Path $quartoExe) { `
            $foundPath = $path ; `
            Write-Host "✅ Found Quarto at: $path" ; `
            Write-Host "   Full path: $quartoExe" ; `
            $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
            if ($mach -notmatch [regex]::Escape($path)) { `
                [Environment]::SetEnvironmentVariable('PATH', ($path + ';' + $mach), 'Machine') ; `
                Write-Host "   ✅ Added $path to PATH" ; `
            } else { `
                Write-Host "   ℹ️  $path already in PATH" ; `
            } ; `
            $quartoFound = $true ; `
            break ; `
        } else { `
            Write-Host "   ❌ Not found: $quartoExe" ; `
        } ; `
    } ; `
    if (-not $quartoFound) { `
        Write-Host '❌ Quarto not found in any of the checked locations:' ; `
        foreach ($path in $quartoPaths) { Write-Host "  - $path" } ; `
        throw 'Quarto installation not found in any expected location' ; `
    } ; `
    Write-Host "🎯 Using Quarto from: $foundPath" ; `
    Write-Host 'Testing Quarto installation...' ; `
    quarto --version

# ------------------------------------------------------------
# PHASE 5: Python 3.13.1 + packages
# ------------------------------------------------------------
RUN Write-Host 'Installing Python 3.13.1...' ; `
    choco install -y python --version=3.13.1 --no-progress ; `
    Write-Host 'Searching for Python installation...' ; `
    $pythonFound = $false ; `
    $pythonPaths = @() ; `
    # Check common Python installation locations
    $pythonPaths += 'C:\Python313' ; `
    $pythonPaths += 'C:\Python313\Scripts' ; `
    $pythonPaths += 'C:\Program Files\Python313' ; `
    $pythonPaths += 'C:\Program Files\Python313\Scripts' ; `
    $pythonPaths += 'C:\Program Files (x86)\Python313' ; `
    $pythonPaths += 'C:\Program Files (x86)\Python313\Scripts' ; `
    $pythonPaths += 'C:\Users\ContainerAdministrator\AppData\Local\Programs\Python\Python313' ; `
    $pythonPaths += 'C:\Users\ContainerAdministrator\AppData\Local\Programs\Python\Python313\Scripts' ; `
    # Also check via Get-Command
    $py = (Get-Command python.exe -All -ErrorAction SilentlyContinue | Select-Object -First 1).Path ; `
    if ($py) { `
        $pyDir = Split-Path $py -Parent ; `
        $scripts = Join-Path $pyDir 'Scripts' ; `
        $pythonPaths += $pyDir ; `
        $pythonPaths += $scripts ; `
    } ; `
    Write-Host 'Checking for Python installation in the following locations:' ; `
    foreach ($path in $pythonPaths) { Write-Host "  - $path" } ; `
    $foundPythonPath = $null ; `
    $foundScriptsPath = $null ; `
    foreach ($path in $pythonPaths) { `
        if (Test-Path $path) { `
            if ($path -like '*Scripts*') { `
                $foundScriptsPath = $path ; `
                Write-Host "✅ Found Python Scripts at: $path" ; `
            } else { `
                $foundPythonPath = $path ; `
                Write-Host "✅ Found Python at: $path" ; `
            } ; `
        } else { `
            Write-Host "   ❌ Not found: $path" ; `
        } ; `
    } ; `
    if (-not $foundPythonPath) { `
        Write-Host '❌ Python installation not found in expected locations' ; `
        throw 'Python installation not found' ; `
    } ; `
    Write-Host "🎯 Using Python from: $foundPythonPath" ; `
    if ($foundScriptsPath) { Write-Host "🎯 Using Python Scripts from: $foundScriptsPath" } ; `
    $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
    foreach ($p in @($foundPythonPath, $foundScriptsPath)) { `
        if ($p -and ($mach -notmatch [regex]::Escape($p))) { `
            $mach = "$p;$mach" ; `
            Write-Host "   ✅ Added $p to PATH" ; `
        } else { `
            Write-Host "   ℹ️  $p already in PATH" ; `
        } ; `
    } ; `
    [Environment]::SetEnvironmentVariable('PATH', $mach, 'Machine') ; `
    Write-Host 'Testing Python installation...' ; `
    python --version ; `
    python -m pip --version ; `
    python -m pip install --upgrade pip ; `
    python -m pip install -r C:/temp/requirements.txt

# ------------------------------------------------------------
# PHASE 6: Ghostscript (Chocolatey)
# ------------------------------------------------------------
RUN Write-Host 'Installing Ghostscript...' ; `
    choco install -y ghostscript --no-progress ; `
    Write-Host 'Searching for Ghostscript installation...' ; `
    $gsFound = $false ; `
    $gsPaths = @() ; `
    # Check common Ghostscript installation locations
    $gsPaths += 'C:\Program Files\gs' ; `
    $gsPaths += 'C:\Program Files (x86)\gs' ; `
    $gsPaths += 'C:\gs' ; `
    # Also check via Get-Command
    $gs = (Get-Command gswin64c.exe -All -ErrorAction SilentlyContinue | Select-Object -First 1).Path ; `
    if ($gs) { `
        $gsBin = Split-Path $gs -Parent ; `
        $gsPaths += $gsBin ; `
    } ; `
    Write-Host 'Checking for Ghostscript installation in the following locations:' ; `
    foreach ($path in $gsPaths) { Write-Host "  - $path" } ; `
    $foundGsPath = $null ; `
    foreach ($path in $gsPaths) { `
        if (Test-Path $path) { `
            $gsBin = Join-Path $path 'bin' ; `
            if (Test-Path (Join-Path $gsBin 'gswin64c.exe')) { `
                $foundGsPath = $gsBin ; `
                Write-Host "✅ Found Ghostscript at: $gsBin" ; `
                Write-Host "   Full path: $(Join-Path $gsBin 'gswin64c.exe')" ; `
                $gsFound = $true ; `
                break ; `
            } ; `
        } ; `
    } ; `
    if (-not $gsFound) { `
        Write-Host '❌ Ghostscript not found in expected locations' ; `
        throw 'Ghostscript installation not found' ; `
    } ; `
    Write-Host "🎯 Using Ghostscript from: $foundGsPath" ; `
    $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
    if ($mach -notmatch [regex]::Escape($foundGsPath)) { `
        [Environment]::SetEnvironmentVariable('PATH', "$foundGsPath;$mach", 'Machine') ; `
        Write-Host "   ✅ Added $foundGsPath to PATH" ; `
    } else { `
        Write-Host "   ℹ️  $foundGsPath already in PATH" ; `
    } ; `
    Write-Host 'Testing Ghostscript installation...' ; `
    Get-Command gswin64c.exe ; `
    gswin64c.exe -h | Select-String -Pattern '^GPL Ghostscript' -SimpleMatch

# ------------------------------------------------------------
# PHASE 7: Inkscape (Chocolatey)
# ------------------------------------------------------------
RUN Write-Host 'Installing Inkscape...' ; `
    choco install -y inkscape --no-progress ; `
    Write-Host 'Searching for Inkscape installation...' ; `
    $inkFound = $false ; `
    $inkPaths = @() ; `
    # Check common Inkscape installation locations
    $inkPaths += 'C:\Program Files\Inkscape\bin' ; `
    $inkPaths += 'C:\Program Files (x86)\Inkscape\bin' ; `
    $inkPaths += 'C:\Inkscape\bin' ; `
    # Also check via Get-Command
    $ink = (Get-Command inkscape.exe -All -ErrorAction SilentlyContinue | Select-Object -First 1).Path ; `
    if ($ink) { `
        $inkBin = Split-Path $ink -Parent ; `
        $inkPaths += $inkBin ; `
    } ; `
    Write-Host 'Checking for Inkscape installation in the following locations:' ; `
    foreach ($path in $inkPaths) { Write-Host "  - $path" } ; `
    $foundInkPath = $null ; `
    foreach ($path in $inkPaths) { `
        if (Test-Path $path) { `
            if (Test-Path (Join-Path $path 'inkscape.exe')) { `
                $foundInkPath = $path ; `
                Write-Host "✅ Found Inkscape at: $path" ; `
                Write-Host "   Full path: $(Join-Path $path 'inkscape.exe')" ; `
                $inkFound = $true ; `
                break ; `
            } ; `
        } ; `
    } ; `
    if (-not $inkFound) { `
        Write-Host '❌ Inkscape not found in expected locations' ; `
        throw 'Inkscape installation not found' ; `
    } ; `
    Write-Host "🎯 Using Inkscape from: $foundInkPath" ; `
    $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
    if ($mach -notmatch [regex]::Escape($foundInkPath)) { `
        [Environment]::SetEnvironmentVariable('PATH', "$foundInkPath;$mach", 'Machine') ; `
        Write-Host "   ✅ Added $foundInkPath to PATH" ; `
    } else { `
        Write-Host "   ℹ️  $foundInkPath already in PATH" ; `
    } ; `
    Write-Host 'Testing Inkscape installation...' ; `
    Get-Command inkscape.exe ; `
    inkscape --version

# ------------------------------------------------------------
# PHASE 8: TeX Live 2025 (pinned) + packages
# ------------------------------------------------------------
RUN Write-Host 'Installing TeX Live 2025 snapshot...' ; `
    $Repo      = 'https://ftp.tug.org/historic/systems/texlive/2025/tlnet-final' ; `
    $Installer = 'C:\temp\install-tl-windows.exe' ; `
    Invoke-WebRequest -Uri 'https://mirror.ctan.org/systems/texlive/tlnet/install-tl-windows.exe' -OutFile $Installer -TimeoutSec 1800 ; `
    'selected_scheme scheme-infraonly' | Set-Content -Path 'C:\temp\texlive.profile' -Encoding ASCII ; `
    'tlpdbopt_install_docfiles 0' | Add-Content -Path 'C:\temp\texlive.profile' -Encoding ASCII ; `
    'tlpdbopt_install_srcfiles 0' | Add-Content -Path 'C:\temp\texlive.profile' -Encoding ASCII ; `
    'TEXDIR C:/texlive' | Add-Content -Path 'C:\temp\texlive.profile' -Encoding ASCII ; `
    'TEXMFCONFIG C:/texlive/texmf-config' | Add-Content -Path 'C:\temp\texlive.profile' -Encoding ASCII ; `
    'TEXMFHOME C:/texlive/texmf-home' | Add-Content -Path 'C:\temp\texlive.profile' -Encoding ASCII ; `
    'TEXMFLOCAL C:/texlive/texmf-local' | Add-Content -Path 'C:\temp\texlive.profile' -Encoding ASCII ; `
    'TEXMFSYSCONFIG C:/texlive/texmf-config' | Add-Content -Path 'C:\temp\texlive.profile' -Encoding ASCII ; `
    'TEXMFSYSVAR C:/texlive/texmf-var' | Add-Content -Path 'C:\temp\texlive.profile' -Encoding ASCII ; `
    'TEXMFVAR C:/texlive/texmf-var' | Add-Content -Path 'C:\temp\texlive.profile' -Encoding ASCII ; `
    Start-Process -FilePath $Installer -ArgumentList '-repository', $Repo, '-profile', 'C:\temp\texlive.profile' -Wait -NoNewWindow ; `
    Write-Host 'Searching for TeX Live installation...' ; `
    $texFound = $false ; `
    $texPaths = @() ; `
    # Check common TeX Live installation locations
    $texPaths += 'C:\texlive\bin\windows' ; `
    $texPaths += 'C:\Program Files\texlive\bin\windows' ; `
    $texPaths += 'C:\Program Files (x86)\texlive\bin\windows' ; `
    # Also check via Get-Command
    $tex = (Get-Command lualatex.exe -All -ErrorAction SilentlyContinue | Select-Object -First 1).Path ; `
    if ($tex) { `
        $texBin = Split-Path $tex -Parent ; `
        $texPaths += $texBin ; `
    } ; `
    Write-Host 'Checking for TeX Live installation in the following locations:' ; `
    foreach ($path in $texPaths) { Write-Host "  - $path" } ; `
    $foundTexPath = $null ; `
    foreach ($path in $texPaths) { `
        if (Test-Path $path) { `
            if (Test-Path (Join-Path $path 'lualatex.exe')) { `
                $foundTexPath = $path ; `
                Write-Host "✅ Found TeX Live at: $path" ; `
                Write-Host "   Full path: $(Join-Path $path 'lualatex.exe')" ; `
                $texFound = $true ; `
                break ; `
            } ; `
        } ; `
    } ; `
    if (-not $texFound) { `
        Write-Host '❌ TeX Live not found in expected locations' ; `
        throw 'TeX Live installation not found' ; `
    } ; `
    Write-Host "🎯 Using TeX Live from: $foundTexPath" ; `
    $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
    if ($mach -notmatch [regex]::Escape($foundTexPath)) { `
        [Environment]::SetEnvironmentVariable('PATH', "$foundTexPath;$mach", 'Machine') ; `
        Write-Host "   ✅ Added $foundTexPath to PATH" ; `
    } else { `
        Write-Host "   ℹ️  $foundTexPath already in PATH" ; `
    } ; `
    Write-Host 'Testing TeX Live installation...' ; `
    & "$foundTexPath\tlmgr.bat" version

RUN $tlmgr = 'C:\texlive\bin\windows\tlmgr.bat' ; `
    $pkgs = Get-Content 'C:\temp\tl_packages' | Where-Object { $_.Trim() -ne '' -and -not $_.Trim().StartsWith('#') } ; `
    foreach ($p in $pkgs) { Write-Host "Installing TeX package: $p" ; & $tlmgr install $p.Trim() } ; `
    & $tlmgr install pgf pgfplots xcolor amsmath standalone psnfss collection-fontsrecommended ; `
    lualatex --version ; `
    pdflatex --version ; `
    foreach ($f in 'pgf.sty','pgfplots.sty','xcolor.sty','amsmath.sty','standalone.cls') { if (-not (kpsewhich $f)) { Write-Error "Missing $f" -ErrorAction Stop } } ; `
    if (-not (kpsewhich phvr7t.tfm)) { Write-Error "Missing Helvetica tfm (phvr7t.tfm)" -ErrorAction Stop } ; `
    if (-not (kpsewhich t1phv.fd))   { Write-Error "Missing Helvetica font descriptor (t1phv.fd)" -ErrorAction Stop }

# TikZ smoke test (fast)
RUN '\documentclass{standalone}' | Set-Content -Path C:\temp\test_tikz.tex -Encoding ASCII ; `
    '\usepackage{tikz}' | Add-Content -Path C:\temp\test_tikz.tex -Encoding ASCII ; `
    '\usepackage{pgfplots}' | Add-Content -Path C:\temp\test_tikz.tex -Encoding ASCII ; `
    '\usepackage{amsmath}' | Add-Content -Path C:\temp\test_tikz.tex -Encoding ASCII ; `
    '\usepackage{xcolor}' | Add-Content -Path C:\temp\test_tikz.tex -Encoding ASCII ; `
    '\usepackage[T1]{fontenc}' | Add-Content -Path C:\temp\test_tikz.tex -Encoding ASCII ; `
    '\usetikzlibrary{positioning,calc}' | Add-Content -Path C:\temp\test_tikz.tex -Encoding ASCII ; `
    '\begin{document}' | Add-Content -Path C:\temp\test_tikz.tex -Encoding ASCII ; `
    '\begin{tikzpicture}[font=\small\usefont{T1}{phv}{m}{n}]' | Add-Content -Path C:\temp\test_tikz.tex -Encoding ASCII ; `
    '\node[draw, fill=blue!20] at (0,0) {TikZ Test};' | Add-Content -Path C:\temp\test_tikz.tex -Encoding ASCII ; `
    '\node[draw, fill=red!20]  at (2,0) {Success};' | Add-Content -Path C:\temp\test_tikz.tex -Encoding ASCII ; `
    '\draw[->] (0.8,0) -- (1.2,0);' | Add-Content -Path C:\temp\test_tikz.tex -Encoding ASCII ; `
    '\end{tikzpicture}' | Add-Content -Path C:\temp\test_tikz.tex -Encoding ASCII ; `
    '\end{document}' | Add-Content -Path C:\temp\test_tikz.tex -Encoding ASCII ; `
    Push-Location C:\temp ; `
    lualatex -interaction=nonstopmode test_tikz.tex ; `
    if (-not (Test-Path C:\temp\test_tikz.pdf)) { throw 'TikZ smoke test failed' } ; `
    Remove-Item C:\temp\test_tikz.* -Force -ErrorAction SilentlyContinue ; `
    Pop-Location

# ------------------------------------------------------------
# PHASE 9: R 4.3.2 + packages
# ------------------------------------------------------------
RUN choco install -y r --version=4.3.2 --no-progress ; `
    R --version | Select-Object -First 1

RUN Rscript -e "options(repos=c(CRAN='https://cran.rstudio.com')); dir.create(Sys.getenv('R_LIBS_USER'), recursive=TRUE, showWarnings=FALSE); .libPaths(Sys.getenv('R_LIBS_USER')); install.packages('remotes'); if (file.exists('C:/temp/install_packages.R')) source('C:/temp/install_packages.R') else install.packages(c('rmarkdown','knitr','ggplot2')); for (p in c('rmarkdown','knitr')) if (!require(p, character.only=TRUE, quietly=TRUE)) stop('missing: ', p)" ; `
    Write-Host 'Verifying R packages...' ; `
    Rscript C:/temp/verify_r_packages.R

# ------------------------------------------------------------
# PHASE 10: Cleanup
# ------------------------------------------------------------
RUN Write-Host 'Cleaning temp files...' ; `
    Remove-Item C:/temp/requirements.txt -ErrorAction SilentlyContinue ; `
    Remove-Item C:/temp/install_packages.R -ErrorAction SilentlyContinue ; `
    Remove-Item C:/temp/verify_r_packages.R -ErrorAction SilentlyContinue ; `
    Remove-Item C:/temp/tl_packages -ErrorAction SilentlyContinue ; `
    Remove-Item C:/temp/requirements/ -Recurse -Force -ErrorAction SilentlyContinue

# ------------------------------------------------------------
# Final checks and metadata
# ------------------------------------------------------------
WORKDIR C:/workspace
RUN quarto --version ; python --version ; R --version ; lualatex --version ; Write-Host 'Windows container build completed successfully ✅'
