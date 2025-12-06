#!/bin/bash
# Fix architecture mismatches in venv
# This script reinstalls packages that may have architecture issues

if [ ! -f ".venv/bin/activate" ]; then
    echo "❌ Error: .venv not found. Run this from the project root."
    exit 1
fi

echo "🔍 Checking venv architecture..."
source .venv/bin/activate

PYTHON_ARCH=$(python -c "import platform; print(platform.machine())")
SYSTEM_ARCH=$(uname -m)

echo "Python architecture: $PYTHON_ARCH"
echo "System architecture: $SYSTEM_ARCH"

if [ "$PYTHON_ARCH" != "$SYSTEM_ARCH" ]; then
    echo "⚠️  Architecture mismatch detected!"
    echo "   Reinstalling problematic packages..."
    pip install --force-reinstall --no-cache-dir rpds jsonschema referencing
    echo "✅ Packages reinstalled"
else
    echo "✅ Architecture match - no action needed"
fi
