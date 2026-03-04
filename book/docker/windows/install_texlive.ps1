param(
    [string]$TexLiveRoot = 'C:\texlive',
    [string]$TexInstallDir = 'C:\texlive-install'
)

$ErrorActionPreference = 'Continue'

$mirrors = @(
    'https://ctan.math.illinois.edu/systems/texlive/tlnet',
    'https://mirrors.mit.edu/CTAN/systems/texlive/tlnet',
    'https://mirror.ctan.org/systems/texlive/tlnet'
)

Write-Host '=== STARTING TEX LIVE INSTALLATION ==='
Write-Host '📥 Downloading install-tl...'
$installTlZip = 'C:\temp\install-tl.zip'
Invoke-WebRequest -Uri 'https://mirror.ctan.org/systems/texlive/tlnet/install-tl.zip' `
    -OutFile $installTlZip -UseBasicParsing
Write-Host '📦 Extracting install-tl...'
Expand-Archive -Path $installTlZip -DestinationPath $TexInstallDir -Force
Remove-Item $installTlZip -Force

$installTlDir = Get-ChildItem $TexInstallDir -Directory |
    Where-Object { $_.Name -match '^install-tl' } |
    Select-Object -First 1
Write-Host "📁 install-tl directory: $($installTlDir.FullName)"

$installed = $false
foreach ($mirror in $mirrors) {
    Write-Host "🔄 Trying mirror: $mirror"
    $profile = 'C:\temp\texlive.profile'
    @(
        "selected_scheme scheme-basic",
        "TEXDIR $TexLiveRoot",
        "TEXMFLOCAL $TexLiveRoot/texmf-local",
        "TEXMFSYSCONFIG $TexLiveRoot/texmf-config",
        "TEXMFSYSVAR $TexLiveRoot/texmf-var",
        "instopt_adjustrepo 0"
    ) | Set-Content $profile
    $env:TEXLIVE_INSTALL_NO_WELCOME = '1'
    $batPath = Join-Path $installTlDir.FullName 'install-tl-windows.bat'
    cmd /c """$batPath"" -no-gui -profile ""$profile"" -repository $mirror"
    if ($LASTEXITCODE -eq 0) {
        $installed = $true
        Write-Host "✅ TeX Live installed from $mirror"
        break
    }
    Write-Host "⚠️ Mirror $mirror failed (exit $LASTEXITCODE), trying next..."
}

Remove-Item $TexInstallDir -Recurse -Force -ErrorAction SilentlyContinue

if (-not $installed) {
    Write-Host '❌ TeX Live installation failed on all mirrors'
    exit 1
}

Write-Host '🔍 Finding TeX Live bin directory...'
Write-Host "  TexLiveRoot contents:"
Get-ChildItem $TexLiveRoot -Directory | ForEach-Object { Write-Host "    $_" }

# Strategy 1: look for a year-numbered directory (e.g. 2025)
$texYearDir = Get-ChildItem $TexLiveRoot -Directory |
    Where-Object { $_.Name -match '^\d{4}$' } |
    Sort-Object Name -Descending |
    Select-Object -First 1

if ($texYearDir) {
    $texLiveBin = Join-Path $texYearDir.FullName 'bin\windows'
} else {
    # Strategy 2: search recursively for tlmgr.bat
    Write-Host '  ⚠️ No year directory found, searching recursively for tlmgr.bat...'
    $tlmgr = Get-ChildItem $TexLiveRoot -Recurse -Filter 'tlmgr.bat' -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($tlmgr) {
        $texLiveBin = $tlmgr.DirectoryName
    } else {
        Write-Host '❌ Cannot find TeX Live bin directory'
        exit 1
    }
}

if (-not (Test-Path $texLiveBin)) {
    Write-Host "❌ TeX Live bin directory does not exist: $texLiveBin"
    exit 1
}
Write-Host "📁 TeX Live bin: $texLiveBin"

# Create a stable symlink so ENV PATH in Docker can use C:\texlive\bin\windows
$stableBin = Join-Path $TexLiveRoot 'bin\windows'
if ($texLiveBin -ne $stableBin -and -not (Test-Path $stableBin)) {
    Write-Host "🔗 Creating stable path: $stableBin -> $texLiveBin"
    New-Item -ItemType Directory -Force -Path (Join-Path $TexLiveRoot 'bin') | Out-Null
    cmd /c mklink /J "$stableBin" "$texLiveBin"
    if (Test-Path $stableBin) {
        Write-Host "✅ Stable symlink created"
    } else {
        Write-Host "⚠️ Symlink failed, using discovered path directly"
    }
}

[Environment]::SetEnvironmentVariable('PATH', ($stableBin + ';' + [Environment]::GetEnvironmentVariable('PATH', 'Machine')), 'Machine')
Write-Host '✅ PATH updated'

Write-Host '🔧 Pinning tlmgr repository to stable mirror...'
$tlmgrMirror = 'https://ctan.math.illinois.edu/systems/texlive/tlnet'
& "$texLiveBin\tlmgr.bat" option repository $tlmgrMirror
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ tlmgr repository: $tlmgrMirror"
} else {
    Write-Host '⚠️ Could not pin tlmgr repository, continuing with defaults'
}

Write-Host '📋 Reading collections from tl_packages...'
if (Test-Path 'C:\temp\tl_packages') {
    $collections = Get-Content 'C:\temp\tl_packages' |
        Where-Object { $_.Trim() -ne '' -and -not $_.Trim().StartsWith('#') }
    Write-Host "📦 Found $($collections.Count) collections to install"
    $i = 1
    foreach ($collection in $collections) {
        Write-Host "📦 [$i/$($collections.Count)] Installing $collection..."
        & "$texLiveBin\tlmgr.bat" install $collection
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $collection installed successfully"
        } else {
            Write-Host "⚠️ Failed to install $collection, continuing..."
        }
        $i++
    }
    Write-Host '✅ Collection installation complete'
} else {
    Write-Host '⚠️ No tl_packages file found, skipping collection installation'
}

Write-Host '🔄 Updating tlmgr...'
& "$texLiveBin\tlmgr.bat" update --self --all
if ($LASTEXITCODE -ne 0) {
    Write-Host '⚠️ tlmgr update returned non-zero, continuing...'
} else {
    Write-Host '✅ tlmgr updated'
}

Write-Host '🔍 Verifying lualatex installation...'
$lualatexPath = Join-Path $texLiveBin 'lualatex.exe'
if (-not (Test-Path $lualatexPath)) {
    Write-Host "❌ lualatex.exe not found at: $lualatexPath"
    Write-Host "  Contents of bin dir:"
    Get-ChildItem $texLiveBin -ErrorAction SilentlyContinue | Select-Object -First 20 | ForEach-Object { Write-Host "    $_" }
    exit 1
}
& $lualatexPath --version
Write-Host '✅ TeX Live installation verified'
