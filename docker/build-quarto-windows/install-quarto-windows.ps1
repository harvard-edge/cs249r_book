<#
    .Synopsis
    Install Quarto release through Scoop package manager
    .Description
    Install Scoop then add r-bucket, and install Quarto from there.
    .Example
    install-quarto-windows.ps1
    install-quarto-windows.ps1 1.7.31
    .Notes
    On Windows, it may be required to enable this script by setting the
    execution policy for the user. You can do this by issuing the following PowerShell
    command:
    PS C:\> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    For more information on Execution Policies: 
    https://go.microsoft.com/fwlink/?LinkID=135170
#>

param(
    [string]$version = ""
)

Write-Host "Installing Quarto via Scoop..."

# Set execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

# Install Scoop
Write-Host "Installing Scoop package manager..."
Invoke-WebRequest -useb get.scoop.sh -outfile 'install.ps1'
& .\install.ps1 -RunAsAdmin

# Add Scoop shims to PATH
$scoopShims = Join-Path (Resolve-Path ~).Path "scoop\shims"
$mach = [Environment]::GetEnvironmentVariable('PATH','Machine')
[Environment]::SetEnvironmentVariable('PATH', ($scoopShims + ';' + $mach), 'Machine')
Write-Host "✅ Added Scoop shims to PATH: $scoopShims"

# Add r-bucket
Write-Host "Adding r-bucket..."
scoop bucket add r-bucket https://github.com/cderv/r-bucket.git

# Install Quarto
if ([string]::IsNullOrEmpty($version)) {
    Write-Host "Installing latest Quarto..."
    scoop install quarto
} elseif ($version -eq 'pre-release') {
    Write-Host "Installing Quarto pre-release..."
    Invoke-Expression -Command "scoop install quarto-prerelease"
} else {
    Write-Host "Installing Quarto version $version..."
    Invoke-Expression -Command "scoop install quarto@$version"
}

Write-Host "✅ Quarto installation completed!"
