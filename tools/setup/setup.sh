#!/bin/bash

# MLSysBook Setup Script
# This script sets up the development environment for new contributors

set -e  # Exit on any error

echo "🚀 Setting up MLSysBook development environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $python_version detected"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install pre-commit hooks
echo "🔗 Installing pre-commit hooks..."
pre-commit install

echo ""
echo "🎉 Setup complete! Your development environment is ready."
echo ""
echo "📋 Next steps:"
echo "  1. Activate the virtual environment: source .venv/bin/activate"
echo "  2. Run pre-commit checks: pre-commit run --all-files"
echo "  3. Start developing!"
echo ""
echo "📚 For more information, see DEPENDENCIES.md" 