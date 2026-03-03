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
$texYearDir = Get-ChildItem $TexLiveRoot -Directory |
    Where-Object { $_.Name -match '^\d{4}$' } |
    Sort-Object Name -Descending |
    Select-Object -First 1
$texLiveBin = Join-Path $texYearDir.FullName 'bin\windows'
Write-Host "📁 TeX Live bin: $texLiveBin"

[Environment]::SetEnvironmentVariable('PATH', ($texLiveBin + ';' + [Environment]::GetEnvironmentVariable('PATH', 'Machine')), 'Machine')
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
Write-Host '✅ tlmgr updated'

Write-Host '🔍 Verifying lualatex installation...'
& "$texLiveBin\lualatex.exe" --version
Write-Host '✅ TeX Live installation verified'
