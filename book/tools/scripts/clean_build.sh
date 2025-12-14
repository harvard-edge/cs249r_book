#!/bin/bash

# Clean Build Artifacts Script
# This script removes build artifacts that might cause the "key:xxx" error

echo "🧹 Cleaning build artifacts..."

# Remove generated TeX files
echo "📄 Removing generated .tex files..."
rm -f quarto/*.tex quarto/*.aux quarto/*.log

# Remove log files from content directories
echo "📋 Removing log files..."
find quarto/contents -name "*.log" -delete 2>/dev/null || true

# Remove build directories
echo "📁 Removing build directories..."
rm -rf quarto/_book quarto/build build

# Remove Quarto cache
echo "🗂️ Removing Quarto cache..."
find . -name ".quarto" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove any remaining temporary files
echo "🗑️ Removing temporary files..."
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*~" -delete 2>/dev/null || true

echo "✅ Build artifacts cleaned!"
echo "💡 You can now try building again."
