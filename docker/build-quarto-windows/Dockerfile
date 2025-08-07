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
    $qDir = Get-ChildItem -Directory C:\quarto | Where-Object { Test-Path (Join-Path $_.FullName 'bin\quarto.exe') } | Select-Object -First 1 ; `
    if (-not $qDir) { throw 'Quarto bin folder not found under C:\quarto' } ; `
    Write-Host "Quarto located at: $($qDir.FullName)" ; `
    $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
    $qbin = Join-Path $qDir.FullName 'bin' ; `
    if ($mach -notmatch [regex]::Escape($qbin)) { [Environment]::SetEnvironmentVariable('PATH', ($qbin + ';' + $mach), 'Machine') } ; `
    Get-Command quarto ; `
    quarto --version

# ------------------------------------------------------------
# PHASE 5: Python 3.13.1 + packages
# ------------------------------------------------------------
RUN choco install -y python --version=3.13.1 --no-progress ; `
    $py = (Get-Command python.exe -All | Select-Object -First 1).Path ; `
    if (-not $py) { throw 'python.exe not found after installation' } ; `
    $pyDir = Split-Path $py -Parent ; `
    $scripts = Join-Path $pyDir 'Scripts' ; `
    $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
    foreach ($p in @($pyDir,$scripts)) { if ($p -and (Test-Path $p) -and ($mach -notmatch [regex]::Escape($p))) { $mach = "$p;$mach" } } ; `
    [Environment]::SetEnvironmentVariable('PATH', $mach, 'Machine') ; `
    python --version ; `
    python -m pip --version ; `
    python -m pip install --upgrade pip ; `
    python -m pip install -r C:/temp/requirements.txt

# ------------------------------------------------------------
# PHASE 6: Ghostscript (Chocolatey)
# ------------------------------------------------------------
RUN choco install -y ghostscript --no-progress ; `
    $gsDir = Get-ChildItem 'C:\Program Files\gs' -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1 ; `
    if (-not $gsDir) { throw 'Ghostscript folder not found after install' } ; `
    $gsBin = Join-Path $gsDir.FullName 'bin' ; `
    $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
    if ($mach -notmatch [regex]::Escape($gsBin)) { [Environment]::SetEnvironmentVariable('PATH', "$gsBin;$mach", 'Machine') } ; `
    Get-Command gswin64c.exe ; `
    gswin64c.exe -h | Select-String -Pattern '^GPL Ghostscript' -SimpleMatch

# ------------------------------------------------------------
# PHASE 7: Inkscape (Chocolatey)
# ------------------------------------------------------------
RUN choco install -y inkscape --no-progress ; `
    $ink = 'C:\Program Files\Inkscape\bin' ; `
    if (-not (Test-Path $ink)) { throw 'Inkscape bin not found' } ; `
    $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
    if ($mach -notmatch [regex]::Escape($ink)) { [Environment]::SetEnvironmentVariable('PATH', "$ink;$mach", 'Machine') } ; `
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
    $tlbin = 'C:\texlive\bin\windows' ; `
    if (-not (Test-Path $tlbin)) { throw "TeX Live bin path missing: $tlbin" } ; `
    $mach = [Environment]::GetEnvironmentVariable('PATH','Machine') ; `
    if ($mach -notmatch [regex]::Escape($tlbin)) { [Environment]::SetEnvironmentVariable('PATH', "$tlbin;$mach", 'Machine') } ; `
    & "$tlbin\tlmgr.bat" version

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
