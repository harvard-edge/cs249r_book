#!/bin/bash

# Part Key Validation Wrapper
# This script runs the Python validation script and provides a summary

echo "🔍 Checking part keys before build..."
echo "=" * 40

# Run the validation script
python3 scripts/validate_part_keys.py
exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ Key validation passed - safe to build!"
    echo "🚀 You can now run: quarto render"
else
    echo "❌ Key validation failed - fix issues before building"
    echo "💡 Run this script again after fixing the issues"
fi

exit $exit_code
