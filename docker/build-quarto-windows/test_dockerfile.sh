#!/bin/bash

# Test script for Windows Dockerfile validation (bash version)
# Run this before building to catch common issues

echo "🔍 Testing Windows Dockerfile configuration..."

# Check if required files exist
required_files=(
    "tools/dependencies/requirements/"
    "tools/dependencies/requirements-build.txt"
    "tools/dependencies/install_packages.R"
    "tools/dependencies/tl_packages"
    "docker/build-quarto-windows/verify_r_packages.R"
)

echo "📁 Checking required files..."
for file in "${required_files[@]}"; do
    if [ -e "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (MISSING)"
        exit 1
    fi
done

# Check Dockerfile syntax
echo "🐳 Validating Dockerfile syntax..."
dockerfile="docker/build-quarto-windows/Dockerfile"
if [ -f "$dockerfile" ]; then
    issues=()
    
    # Check for proper escape character
    if ! grep -q "^# escape=\`" "$dockerfile"; then
        issues+=("Missing escape character at top")
    fi
    
    # Check for proper SHELL commands (should use full path)
    if grep -q 'SHELL \["pwsh"' "$dockerfile"; then
        issues+=("Using 'pwsh' instead of full path - should use 'C:\\\\Program Files\\\\PowerShell\\\\7\\\\pwsh.exe'")
    fi
    
    # Check for proper PowerShell commands
    if grep -q 'Start-Process.*-Wait' "$dockerfile" && ! grep -q 'Start-Process.*-Wait.*-NoNewWindow' "$dockerfile"; then
        issues+=("Start-Process should include -NoNewWindow for container builds")
    fi
    
    # Check for comment filtering in tl_packages
    if ! grep -q "StartsWith('#')" "$dockerfile"; then
        issues+=("Missing comment filtering for tl_packages")
    fi
    
    if [ ${#issues[@]} -eq 0 ]; then
        echo "  ✅ Dockerfile syntax looks good"
    else
        echo "  ⚠️  Potential issues found:"
        for issue in "${issues[@]}"; do
            echo "    - $issue"
        done
    fi
else
    echo "  ❌ Dockerfile not found"
    exit 1
fi

# Check tl_packages content
echo "📦 Checking TeX Live packages..."
tl_packages="tools/dependencies/tl_packages"
if [ -f "$tl_packages" ]; then
    package_count=$(grep -v '^#' "$tl_packages" | grep -v '^$' | wc -l)
    echo "  ✅ Found $package_count TeX packages to install"
else
    echo "  ❌ tl_packages file missing"
fi

# Check requirements
echo "🐍 Checking Python requirements..."
requirements="tools/dependencies/requirements-build.txt"
if [ -f "$requirements" ]; then
    echo "  ✅ Requirements file found"
else
    echo "  ❌ Requirements file missing"
fi

echo "✅ Dockerfile validation complete!"
echo ""
echo "To build the container:"
echo "  docker build -f docker/build-quarto-windows/Dockerfile -t mlsysbook-windows ."
