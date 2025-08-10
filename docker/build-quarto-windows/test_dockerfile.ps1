#!/usr/bin/env pwsh

# Test script for Windows Dockerfile validation
# Run this before building to catch common issues

# Variables
$headline = "🚀 Testing Dockerfile: Windows"
$dockerfile = "docker/build-quarto-windows/Dockerfile"
$image_name = "mlsysbook-windows-test"
$container_name = "mlsysbook-windows-test-container"

Write-Host $headline -ForegroundColor Green

# Check if required files exist
$requiredFiles = @(
    "tools/dependencies/requirements/",
    "tools/dependencies/requirements-build.txt", 
    "tools/dependencies/install_packages.R",
    "tools/dependencies/tl_packages",
    "docker/build-quarto-windows/verify_r_packages.R"
)

Write-Host "📁 Checking required files..." -ForegroundColor Yellow
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file (MISSING)" -ForegroundColor Red
        exit 1
    }
}

# Check Dockerfile syntax
Write-Host "🐳 Validating Dockerfile syntax..." -ForegroundColor Yellow
if (Test-Path $dockerfile) {
    $content = Get-Content $dockerfile -Raw
    
    # Check for common Windows container issues
    $issues = @()
    
    # Check for proper escape character
    if ($content -notmatch "# escape=`") {
        $issues += "Missing escape character at top"
    }
    
    # Check for proper SHELL commands
    if ($content -match 'SHELL \["pwsh"') {
        $issues += "Using 'pwsh' instead of full path - should use 'C:\\Program Files\\PowerShell\\7\\pwsh.exe'"
    }
    
    # Check for proper line continuation
    if ($content -match '`\s*$') {
        $issues += "Trailing backticks found - should be removed"
    }
    
    # Check for proper PowerShell commands
    if ($content -match 'Start-Process.*-Wait(?!.*-NoNewWindow)') {
        $issues += "Start-Process should include -NoNewWindow for container builds"
    }
    
    if ($issues.Count -eq 0) {
        Write-Host "  ✅ Dockerfile syntax looks good" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Potential issues found:" -ForegroundColor Yellow
        foreach ($issue in $issues) {
            Write-Host "    - $issue" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  ❌ Dockerfile not found" -ForegroundColor Red
    exit 1
}

# Check tl_packages content
Write-Host "📦 Checking TeX Live packages..." -ForegroundColor Yellow
$tlPackages = "tools/dependencies/tl_packages"
if (Test-Path $tlPackages) {
    $packages = Get-Content $tlPackages | Where-Object { $_.Trim() -ne '' -and -not $_.Trim().StartsWith('#') }
    Write-Host "  ✅ Found $($packages.Count) TeX packages to install" -ForegroundColor Green
} else {
    Write-Host "  ❌ tl_packages file missing" -ForegroundColor Red
}

# Check requirements
Write-Host "🐍 Checking Python requirements..." -ForegroundColor Yellow
$requirements = "tools/dependencies/requirements-build.txt"
if (Test-Path $requirements) {
    Write-Host "  ✅ Requirements file found" -ForegroundColor Green
} else {
    Write-Host "  ❌ Requirements file missing" -ForegroundColor Red
}

Write-Host "✅ Dockerfile validation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To build the container:" -ForegroundColor Cyan
Write-Host "  docker build -f docker/build-quarto-windows/Dockerfile -t mlsysbook-windows ." -ForegroundColor White
