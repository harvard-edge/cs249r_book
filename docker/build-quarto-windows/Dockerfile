# MLSysBook Windows Quarto Build Container (Windows Server 2022)
# - PowerShell 7 via ZIP (no MSI)
# - Quarto 1.7.31
# - Python 3.13.1 + requirements
# - Ghostscript + Inkscape (Chocolatey)
# - TeX Live pinned to 2025 snapshot + packages from tl_packages
# - R 4.3.2 + packages via install_packages.R
# - Verifications: versions, kpsewhich font files, TikZ smoke test

FROM mcr.microsoft.com/windows/server:ltsc2022

# Use PowerShell 5.1 initially (before PowerShell 7 installation)
SHELL ["powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass", "-Command"]

# ------------------------------------------------------------
# PHASE 0: Base dirs and env
# ------------------------------------------------------------
ENV R_LIBS_USER="C:/r-lib"
ENV QUARTO_LOG_LEVEL="INFO"
ENV PYTHONIOENCODING="utf-8"
ENV LANG="en_US.UTF-8"
ENV LC_ALL="en_US.UTF-8"

RUN powershell.exe -Command "New-Item -ItemType Directory -Force -Path 'C:\temp' | Out-Null; New-Item -ItemType Directory -Force -Path 'C:\r-lib' | Out-Null"

# ------------------------------------------------------------
# PHASE 1: PowerShell 7 (ZIP install, container-safe)
# ------------------------------------------------------------
RUN powershell.exe -Command "Write-Host '🚀 Installing PowerShell 7 (ZIP) ...'; $Url = 'https://github.com/PowerShell/PowerShell/releases/download/v7.4.1/PowerShell-7.4.1-win-x64.zip'; $Zip = 'C:\PowerShell-7.4.1.zip'; Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing; New-Item -ItemType Directory -Force -Path 'C:\Program Files\PowerShell\7' | Out-Null; Expand-Archive -Path $Zip -DestinationPath 'C:\Program Files\PowerShell\7' -Force; Remove-Item $Zip -Force"
ENV PATH="C:\\Program Files\\PowerShell\\7;%PATH%"

# Switch to PowerShell 7 for all subsequent commands
SHELL ["pwsh", "-NoLogo", "-ExecutionPolicy", "Bypass", "-Command"]

RUN & 'C:\Program Files\PowerShell\7\pwsh.exe' -NoLogo -NoProfile -Command `
      '$PSVersionTable.PSVersion; Write-Host ''PowerShell 7 installation verified ✅'''

# ------------------------------------------------------------
# PHASE 2: Chocolatey
# ------------------------------------------------------------
RUN Write-Host '🚀 Installing Chocolatey ...'; `
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; `
    iex ((New-Object Net.WebClient).DownloadString('https://chocolatey.org/install.ps1')); `
    choco --version

# ------------------------------------------------------------
# PHASE 3: Copy dependency files
# ------------------------------------------------------------
# These come from your repo
COPY tools/dependencies/requirements/         C:/temp/requirements/
COPY tools/dependencies/requirements-build.txt C:/temp/requirements.txt
COPY tools/dependencies/install_packages.R     C:/temp/install_packages.R
COPY tools/dependencies/tl_packages            C:/temp/tl_packages
COPY docker/build-quarto-windows/verify_r_packages.R C:/temp/verify_r_packages.R

# ------------------------------------------------------------
# PHASE 4: Quarto 1.7.31
# ------------------------------------------------------------
RUN Write-Host '🚀 Installing Quarto 1.7.31 ...'; `
    Invoke-WebRequest -Uri 'https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.31/quarto-1.7.31-win.zip' -OutFile 'C:\quarto.zip' -TimeoutSec 300; `
    Expand-Archive -Path 'C:\quarto.zip' -DestinationPath 'C:\quarto' -Force; `
    Remove-Item 'C:\quarto.zip' -Force
ENV PATH="C:\\quarto\\quarto-1.7.31\\bin;%PATH%"

RUN quarto --version

# ------------------------------------------------------------
# PHASE 5: Python 3.13.1 + packages
# ------------------------------------------------------------
RUN choco install -y python --version=3.13.1; `
    python --version; `
    python -m pip --version
ENV PATH="C:\\Python313;C:\\Python313\\Scripts;%PATH%"

RUN python -m pip install --upgrade pip; `
    python -m pip install -r C:/temp/requirements.txt

# ------------------------------------------------------------
# PHASE 6: Ghostscript (Chocolatey)
# ------------------------------------------------------------
RUN choco install -y ghostscript; `
    $gsDir = Get-ChildItem 'C:\Program Files\gs' -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1; `
    if ($gsDir) { `
        $bin = Join-Path $gsDir.FullName 'bin'; `
        & cmd /c 'setx /M PATH \"%PATH%;$bin\"' | Out-Null `
    } else { `
        throw 'Ghostscript folder not found after install' `
    }
# New layer to pick up PATH
RUN gswin64c.exe -h | Select-String -Pattern '^GPL Ghostscript'

# ------------------------------------------------------------
# PHASE 7: Inkscape (Chocolatey)
# ------------------------------------------------------------
RUN choco install -y inkscape
ENV PATH="C:\\Program Files\\Inkscape\\bin;%PATH%"

RUN inkscape --version

# ------------------------------------------------------------
# PHASE 8: TeX Live 2025 (pinned) + packages
# ------------------------------------------------------------
# NOTE: We pin to the historic TL 2025 snapshot to mirror CI's texlive_version: 2025.
# If that snapshot URL changes, update the $Repo accordingly.
RUN Write-Host '🚀 Installing TeX Live (2025 snapshot) ...'; `
    $Repo = 'https://ftp.tug.org/historic/systems/texlive/2025/tlnet-final'; `
    $Installer = 'C:\temp\install-tl-windows.exe'; `
    Invoke-WebRequest -Uri 'https://mirror.ctan.org/systems/texlive/tlnet/install-tl-windows.exe' -OutFile $Installer -TimeoutSec 600; `
    $Profile = @'
selected_scheme scheme-infraonly
tlpdbopt_install_docfiles 0
tlpdbopt_install_srcfiles 0
TEXDIR C:/texlive
TEXMFCONFIG C:/texlive/texmf-config
TEXMFHOME C:/texlive/texmf-home
TEXMFLOCAL C:/texlive/texmf-local
TEXMFSYSCONFIG C:/texlive/texmf-config
TEXMFSYSVAR C:/texlive/texmf-var
TEXMFVAR C:/texlive/texmf-var
'@; `
    $Profile | Set-Content -Path 'C:\temp\texlive.profile' -Encoding ASCII; `
    Start-Process -FilePath $Installer -ArgumentList '-repository', $Repo, '-profile', 'C:\temp\texlive.profile' -Wait; `
    & 'C:\texlive\bin\windows\tlmgr.bat' version
ENV PATH="C:\\texlive\\bin\\windows;%PATH%"

# Install collections from tl_packages
RUN $pkgs = Get-Content 'C:\temp\tl_packages' | Where-Object { $_.Trim() -ne '' }; `
    foreach ($p in $pkgs) { `
        & 'C:\texlive\bin\windows\tlmgr.bat' install $p.Trim() `
    }

# TeX checks incl. Helvetica files
RUN lualatex --version; `
    pdflatex --version; `
    foreach ($f in 'pgf.sty','pgfplots.sty','xcolor.sty','amsmath.sty','standalone.cls') { `
        if (-not (kpsewhich $f)) { `
            Write-Error "Missing $f" -ErrorAction Stop `
        } `
    }; `
    if (-not (kpsewhich phvr7t.tfm)) { `
        Write-Error "Missing Helvetica tfm (phvr7t.tfm)" -ErrorAction Stop `
    }; `
    if (-not (kpsewhich t1phv.fd)) { `
        Write-Error "Missing Helvetica font descriptor (t1phv.fd)" -ErrorAction Stop `
    }

# TikZ smoke test (fast)
RUN Set-Content -Path C:\temp\test_tikz.tex -Value @'
\documentclass{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage[T1]{fontenc}
\usetikzlibrary{positioning,calc}
\begin{document}
\begin{tikzpicture}[font=\small\usefont{T1}{phv}{m}{n}]
\node[draw, fill=blue!20] at (0,0) {TikZ Test};
\node[draw, fill=red!20]  at (2,0) {Success};
\draw[->] (0.8,0) -- (1.2,0);
\end{tikzpicture}
\end{document}
'@ -Encoding ASCII; `
    Push-Location C:\temp; `
    lualatex -interaction=nonstopmode test_tikz.tex; `
    if (-not (Test-Path C:\temp\test_tikz.pdf)) { `
        throw 'TikZ smoke test failed' `
    }; `
    Remove-Item C:\temp\test_tikz.* -Force -ErrorAction SilentlyContinue; `
    Pop-Location

# ------------------------------------------------------------
# PHASE 9: R 4.3.2 + packages
# ------------------------------------------------------------
RUN choco install -y r --version=4.3.2; `
    R --version | Select-Object -First 1

RUN Rscript -e "options(repos=c(CRAN='https://cran.rstudio.com')); \
                dir.create(Sys.getenv('R_LIBS_USER'), recursive=TRUE, showWarnings=FALSE); \
                .libPaths(Sys.getenv('R_LIBS_USER')); \
                install.packages('remotes'); \
                if (file.exists('C:/temp/install_packages.R')) source('C:/temp/install_packages.R') else install.packages(c('rmarkdown','knitr','ggplot2')); \
                for (p in c('rmarkdown','knitr')) if (!require(p, character.only=TRUE, quietly=TRUE)) stop('missing: ', p)"

# Optional: your R verification script
RUN Write-Host '🔍 Verifying R packages...'; `
    Rscript C:/temp/verify_r_packages.R

# ------------------------------------------------------------
# PHASE 10: Cleanup
# ------------------------------------------------------------
RUN Write-Host '🧹 Cleaning temp files...'; `
    Remove-Item C:/temp/requirements.txt -ErrorAction SilentlyContinue; `
    Remove-Item C:/temp/install_packages.R -ErrorAction SilentlyContinue; `
    Remove-Item C:/temp/verify_r_packages.R -ErrorAction SilentlyContinue; `
    Remove-Item C:/temp/tl_packages -ErrorAction SilentlyContinue; `
    Remove-Item C:/temp/requirements/ -Recurse -Force -ErrorAction SilentlyContinue

# ------------------------------------------------------------
# Final checks and metadata
# ------------------------------------------------------------
WORKDIR C:/workspace
RUN quarto --version; python --version; R --version; lualatex --version
RUN Write-Host '✅ Windows container build completed successfully'
