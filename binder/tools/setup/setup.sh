#!/bin/bash

# MLSysBook Setup Script
# This script sets up the development environment for new contributors

set -e  # Exit on any error

echo "🚀 Setting up MLSysBook development environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.9+ first."
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
pip install -r tools/dependencies/requirements.txt

# Install pre-commit hooks
echo "🔗 Installing pre-commit hooks..."
pre-commit install

# Check for GitHub CLI
echo "🔍 Checking for GitHub CLI..."
if ! command -v gh &> /dev/null; then
    echo "📦 Installing GitHub CLI..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install gh
        else
            echo "❌ Homebrew is required to install GitHub CLI on macOS"
            echo "   Install Homebrew first: https://brew.sh"
            echo "   Then run: brew install gh"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
        sudo apt update
        sudo apt install gh
    else
        echo "⚠️ Please install GitHub CLI manually: https://cli.github.com/"
    fi
else
    echo "✅ GitHub CLI already installed"
fi

echo ""
echo "🎉 Setup complete! Your development environment is ready."
echo ""
echo "📋 Next steps:"
echo "  1. Activate the virtual environment: source .venv/bin/activate"
echo "  2. Authenticate with GitHub: gh auth login"
echo "  3. Run pre-commit checks: pre-commit run --all-files"
echo "  4. Start developing!"
echo ""
echo "📚 For more information, see DEPENDENCIES.md"
