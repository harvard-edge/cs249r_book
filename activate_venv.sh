#!/bin/bash
# MLSysBook Virtual Environment Activation Script
# Usage: source activate_venv.sh

# Check if we're already in the virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Already in virtual environment: $VIRTUAL_ENV"
    exit 0
fi

# Check if .venv exists
if [[ ! -d ".venv" ]]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv .venv
    echo "✅ Virtual environment created."
fi

# Activate the virtual environment
source .venv/bin/activate

# Verify activation
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
    echo "📦 Python version: $(python --version)"
    echo "📍 Python location: $(which python)"
    echo ""
    echo "💡 To deactivate, run: deactivate"
    echo "💡 To install dependencies, run: pip install -r requirements.txt"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi



