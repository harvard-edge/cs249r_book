#!/bin/bash
# MLSysBook direnv activation script
# Use this to manually activate direnv in Cursor's terminal or any shell session
#
# Usage:
#   source ./activate_direnv.sh
#   # or
#   . ./activate_direnv.sh

echo "🔧 Activating direnv for MLSysBook..."

# Load direnv hook
eval "$(direnv hook bash)"

# Change to project directory to trigger direnv
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "✅ direnv activated!"
echo "🐍 Python: $(python --version 2>/dev/null || echo 'Not available')"
echo "📍 Virtual env: ${VIRTUAL_ENV:-'Not set'}"
echo "📁 Project root: ${PROJECT_ROOT:-'Not set'}"



