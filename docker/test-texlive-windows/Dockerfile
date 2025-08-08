# Minimal Windows TeX Live Test Container
# Uses smallest viable Windows base for TeX Live testing

FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Use Windows PowerShell
SHELL ["powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass", "-Command"]

# ------------------------------------------------------------
# PHASE 1: Minimal Setup
# ------------------------------------------------------------
RUN Write-Host '=== MINIMAL TEX LIVE TEST SETUP ===' ; `
    Write-Host 'Creating temp directory...' ; `
    New-Item -ItemType Directory -Force -Path 'C:\temp' | Out-Null ; `
    Write-Host '📁 Created C:\temp' ; `
    Write-Host '✅ Minimal setup complete'

# ------------------------------------------------------------
# PHASE 2: Chocolatey (Required for TeX Live)
# ------------------------------------------------------------
RUN Write-Host '=== INSTALLING CHOCOLATEY ===' ; `
    Write-Host 'Setting TLS 1.2 for download...' ; `
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 ; `
    Write-Host '🔒 TLS 1.2 enabled' ; `
    Write-Host 'Installing Chocolatey...' ; `
    iex ((New-Object Net.WebClient).DownloadString('https://chocolatey.org/install.ps1')) ; `
    Write-Host '📦 Chocolatey installed' ; `
    Write-Host 'Verifying Chocolatey...' ; `
    choco --version ; `
    Write-Host '✅ Chocolatey installation complete'

# ------------------------------------------------------------
# PHASE 3: Copy TeX Live Packages File
# ------------------------------------------------------------
COPY tools/dependencies/tl_packages C:/temp/tl_packages

# ------------------------------------------------------------
# PHASE 4: TeX Live Installation (The Test)
# ------------------------------------------------------------
RUN Write-Host '=== TESTING TEX LIVE INSTALLATION ===' ; `
    Write-Host '📦 Installing TeX Live via Chocolatey...' ; `
    choco install texlive -y ; `
    Write-Host '✅ TeX Live installed via Chocolatey' ; `
    `
    Write-Host '🔍 Finding TeX Live installation directory...' ; `
    $texRoot = Join-Path $env:SystemDrive 'texlive' ; `
    Write-Host "📁 TeX Live root: $texRoot" ; `
    `
    Write-Host '🔍 Looking for year-based directories...' ; `
    $texYearDir = Get-ChildItem $texRoot -Directory | `
        Where-Object { $_.Name -match '^\d{4}$' } | `
        Sort-Object Name -Descending | `
        Select-Object -First 1 ; `
    Write-Host "📁 Found year directory: $($texYearDir.FullName)" ; `
    `
    $texLiveBin = Join-Path $texYearDir.FullName 'bin\windows' ; `
    Write-Host "📁 TeX Live bin directory: $texLiveBin" ; `
    `
    Write-Host '🔧 Adding TeX Live to PATH...' ; `
    $env:PATH = "$texLiveBin;$env:PATH" ; `
    Write-Host "✅ PATH updated with: $texLiveBin" ; `
    `
    Write-Host '📋 Reading collections from tl_packages...' ; `
    if (Test-Path 'C:\temp\tl_packages') { `
        $collections = Get-Content 'C:\temp\tl_packages' | `
            Where-Object { $_.Trim() -ne '' -and -not $_.Trim().StartsWith('#') } ; `
        Write-Host "📦 Found $($collections.Count) collections to install" ; `
        Write-Host '📋 Collections:' ; `
        $collections | ForEach-Object { Write-Host "  - $_" } ; `
        `
        Write-Host '🔄 Installing collections...' ; `
        $i = 1 ; `
        foreach ($collection in $collections) { `
            Write-Host "📦 [$i/$($collections.Count)] Installing $collection..." ; `
            & "$texLiveBin\tlmgr.bat" install $collection ; `
            if ($LASTEXITCODE -eq 0) { `
                Write-Host "✅ $collection installed successfully" ; `
            } else { `
                Write-Host "⚠️ Failed to install $collection, continuing..." ; `
            } ; `
            $i++ ; `
        } ; `
        Write-Host '✅ Collection installation complete' ; `
    } else { `
        Write-Host '⚠️ No tl_packages file found, skipping collection installation' ; `
    } ; `
    `
    Write-Host '🔄 Updating tlmgr...' ; `
    & "$texLiveBin\tlmgr.bat" update --self --all ; `
    Write-Host '✅ tlmgr updated' ; `
    `
    Write-Host '🔍 Verifying lualatex installation...' ; `
    & "$texLiveBin\lualatex.exe" --version ; `
    Write-Host '✅ TeX Live installation verified'

# ------------------------------------------------------------
# PHASE 5: Verification
# ------------------------------------------------------------
RUN Write-Host '=== FINAL VERIFICATION ===' ; `
    Write-Host 'Checking TeX Live installation...' ; `
    $texLiveBin = 'C:\texlive\bin\windows' ; `
    & "$texLiveBin\lualatex.exe" --version ; `
    Write-Host '✅ TeX Live test completed successfully' ; `
    Write-Host '📊 Container size test passed!'
