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
# PHASE 4: Install Scoop
# ------------------------------------------------------------
RUN Write-Host 'Installing Scoop...' ; `
    Invoke-WebRequest -useb get.scoop.sh -outfile 'C:\temp\install.ps1' ; `
    & 'C:\temp\install.ps1' -RunAsAdmin ; `
    $scoopShims = Join-Path (Resolve-Path ~).Path "scoop\shims" ; `
    $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
    [Environment]::SetEnvironmentVariable('PATH', ($scoopShims + ';' + $mach), 'Machine') ; `
    Write-Host "✅ Added Scoop shims to PATH: $scoopShims" ; `
    scoop bucket add r-bucket https://github.com/cderv/r-bucket.git

# ------------------------------------------------------------
# PHASE 5: Install Quarto
# ------------------------------------------------------------
RUN Write-Host 'Installing Quarto 1.7.31...' ; `
    scoop install quarto@1.7.31

# ------------------------------------------------------------
# PHASE 6: Install Python
# ------------------------------------------------------------
RUN Write-Host 'Installing Python...' ; `
    scoop install main/python

# ------------------------------------------------------------
# PHASE 7: Install Ghostscript
# ------------------------------------------------------------
RUN Write-Host 'Installing Ghostscript...' ; `
    scoop install main/ghostscript

# ------------------------------------------------------------
# PHASE 8: Install Inkscape
# ------------------------------------------------------------
RUN Write-Host 'Installing Inkscape...' ; `
    scoop install extras/inkscape

# ------------------------------------------------------------
# PHASE 9: Install R
# ------------------------------------------------------------
RUN Write-Host 'Installing R...' ; `
    scoop install main/r

# ------------------------------------------------------------
# PHASE 10: Install TeX Live
# ------------------------------------------------------------
RUN Write-Host 'Installing TeX Live 2025...' ; `
    $Repo = 'https://ftp.tug.org/historic/systems/texlive/2025/tlnet-final' ; `
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
    Start-Process -FilePath $Installer -ArgumentList '-repository', $Repo, '-profile', 'C:\temp\texlive.profile' -Wait -NoNewWindow

# ------------------------------------------------------------
# PHASE 11: Discover all installed tools
# ------------------------------------------------------------
RUN Write-Host '=== DISCOVERING ALL INSTALLED TOOLS ===' ; `
    Write-Host 'Searching for Scoop-installed tools...' ; `
    $scoopRoot = (Resolve-Path ~).Path + '\scoop' ; `
    Write-Host "Scoop root: $scoopRoot" ; `
    Write-Host 'Searching for Quarto...' ; `
    $quartoLocation = where.exe quarto 2>$null ; `
    if ($quartoLocation) { `
        $quartoPath = $quartoLocation.Split([Environment]::NewLine)[0] ; `
        $quartoDir = Split-Path $quartoPath -Parent ; `
        Write-Host "✅ Found Quarto at: $quartoDir" ; `
        Write-Host "   Full path: $quartoPath" ; `
    } else { `
        Write-Host '❌ Quarto not found' ; `
    } ; `
    Write-Host 'Searching for Python...' ; `
    $pythonLocation = where.exe python 2>$null ; `
    if ($pythonLocation) { `
        $pythonPath = $pythonLocation.Split([Environment]::NewLine)[0] ; `
        $pythonDir = Split-Path $pythonPath -Parent ; `
        Write-Host "✅ Found Python at: $pythonDir" ; `
        Write-Host "   Full path: $pythonPath" ; `
    } else { `
        Write-Host '❌ Python not found' ; `
    } ; `
    Write-Host 'Searching for Ghostscript...' ; `
    $gsLocation = where.exe gswin64c 2>$null ; `
    if ($gsLocation) { `
        $gsPath = $gsLocation.Split([Environment]::NewLine)[0] ; `
        $gsDir = Split-Path $gsPath -Parent ; `
        Write-Host "✅ Found Ghostscript at: $gsDir" ; `
        Write-Host "   Full path: $gsPath" ; `
    } else { `
        Write-Host '❌ Ghostscript not found' ; `
    } ; `
    Write-Host 'Searching for Inkscape...' ; `
    $inkLocation = where.exe inkscape 2>$null ; `
    if ($inkLocation) { `
        $inkPath = $inkLocation.Split([Environment]::NewLine)[0] ; `
        $inkDir = Split-Path $inkPath -Parent ; `
        Write-Host "✅ Found Inkscape at: $inkDir" ; `
        Write-Host "   Full path: $inkPath" ; `
    } else { `
        Write-Host '❌ Inkscape not found' ; `
    } ; `
    Write-Host 'Searching for TeX Live...' ; `
    $texLocation = where.exe lualatex 2>$null ; `
    if ($texLocation) { `
        $texPath = $texLocation.Split([Environment]::NewLine)[0] ; `
        $texDir = Split-Path $texPath -Parent ; `
        Write-Host "✅ Found TeX Live at: $texDir" ; `
        Write-Host "   Full path: $texPath" ; `
    } else { `
        Write-Host '❌ TeX Live not found' ; `
    } ; `
    Write-Host 'Searching for R...' ; `
    $rLocation = where.exe R 2>$null ; `
    if ($rLocation) { `
        $rPath = $rLocation.Split([Environment]::NewLine)[0] ; `
        $rDir = Split-Path $rPath -Parent ; `
        Write-Host "✅ Found R at: $rDir" ; `
        Write-Host "   Full path: $rPath" ; `
    } else { `
        Write-Host '❌ R not found' ; `
    } ; `
    Write-Host '=== TOOL DISCOVERY COMPLETE ==='

# ------------------------------------------------------------
# PHASE 5: Python 3.13.1 + packages
# ------------------------------------------------------------
RUN Write-Host 'Installing Python 3.13.1...' ; `
    choco install -y python --version=3.13.1 --no-progress ; `
    Write-Host 'Searching for Python installation...' ; `
    # Use where.exe to find python
    Write-Host 'Searching for python using where.exe...' ; `
    $pythonLocation = where.exe python 2>$null ; `
    if ($pythonLocation) { `
        $pythonPath = $pythonLocation.Split([Environment]::NewLine)[0] ; `
        $pythonDir = Split-Path $pythonPath -Parent ; `
        $scriptsDir = Join-Path $pythonDir 'Scripts' ; `
        Write-Host "✅ Found Python at: $pythonDir" ; `
        Write-Host "   Full path: $pythonPath" ; `
        if (Test-Path $scriptsDir) { `
            Write-Host "✅ Found Python Scripts at: $scriptsDir" ; `
        } ; `
        Write-Host "🎯 Using Python from: $pythonDir" ; `
        # Add to PATH if not already there
        $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
        foreach ($p in @($pythonDir, $scriptsDir)) { `
            if ($p -and (Test-Path $p) -and ($mach -notmatch [regex]::Escape($p))) { `
                $mach = "$p;$mach" ; `
                Write-Host "   ✅ Added $p to PATH" ; `
            } ; `
        } ; `
        [Environment]::SetEnvironmentVariable('PATH', $mach, 'Machine') ; `
    } else { `
        Write-Host '❌ Python not found using where.exe' ; `
        throw 'Python installation not found' ; `
    } ; `
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
    # Use where.exe to find gswin64c
    Write-Host 'Searching for gswin64c using where.exe...' ; `
    $gsLocation = where.exe gswin64c 2>$null ; `
    if ($gsLocation) { `
        $gsPath = $gsLocation.Split([Environment]::NewLine)[0] ; `
        $gsDir = Split-Path $gsPath -Parent ; `
        Write-Host "✅ Found Ghostscript at: $gsDir" ; `
        Write-Host "   Full path: $gsPath" ; `
        Write-Host "🎯 Using Ghostscript from: $gsDir" ; `
        # Add to PATH if not already there
        $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
        if ($mach -notmatch [regex]::Escape($gsDir)) { `
            [Environment]::SetEnvironmentVariable('PATH', "$gsDir;$mach", 'Machine') ; `
            Write-Host "   ✅ Added $gsDir to PATH" ; `
        } ; `
    } else { `
        Write-Host '❌ Ghostscript not found using where.exe' ; `
        throw 'Ghostscript installation not found' ; `
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
    # Use where.exe to find inkscape
    Write-Host 'Searching for inkscape using where.exe...' ; `
    $inkLocation = where.exe inkscape 2>$null ; `
    if ($inkLocation) { `
        $inkPath = $inkLocation.Split([Environment]::NewLine)[0] ; `
        $inkDir = Split-Path $inkPath -Parent ; `
        Write-Host "✅ Found Inkscape at: $inkDir" ; `
        Write-Host "   Full path: $inkPath" ; `
        Write-Host "🎯 Using Inkscape from: $inkDir" ; `
        # Add to PATH if not already there
        $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
        if ($mach -notmatch [regex]::Escape($inkDir)) { `
            [Environment]::SetEnvironmentVariable('PATH', "$inkDir;$mach", 'Machine') ; `
            Write-Host "   ✅ Added $inkDir to PATH" ; `
        } ; `
    } else { `
        Write-Host '❌ Inkscape not found using where.exe' ; `
        throw 'Inkscape installation not found' ; `
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
    # Use where.exe to find lualatex
    Write-Host 'Searching for lualatex using where.exe...' ; `
    $texLocation = where.exe lualatex 2>$null ; `
    if ($texLocation) { `
        $texPath = $texLocation.Split([Environment]::NewLine)[0] ; `
        $texDir = Split-Path $texPath -Parent ; `
        Write-Host "✅ Found TeX Live at: $texDir" ; `
        Write-Host "   Full path: $texPath" ; `
        Write-Host "🎯 Using TeX Live from: $texDir" ; `
        # Add to PATH if not already there
        $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
        if ($mach -notmatch [regex]::Escape($texDir)) { `
            [Environment]::SetEnvironmentVariable('PATH', "$texDir;$mach", 'Machine') ; `
            Write-Host "   ✅ Added $texDir to PATH" ; `
        } ; `
    } else { `
        Write-Host '❌ TeX Live not found using where.exe' ; `
        throw 'TeX Live installation not found' ; `
    } ; `
    Write-Host 'Testing TeX Live installation...' ; `
    & "$texDir\tlmgr.bat" version

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
