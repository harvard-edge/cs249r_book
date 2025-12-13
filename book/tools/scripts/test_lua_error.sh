#!/bin/bash

# Test Lua Error Handling Script
# This script tests the inject-parts.lua error handling

echo "🧪 Testing Lua error handling..."

# Create a temporary test file with an invalid key
cat > test_invalid_key.qmd << 'EOF'
---
title: "Test Invalid Key"
format: pdf
---

# Test

\part{key:invalid_key}

Content here.
EOF

echo "📄 Created test file with invalid key 'invalid_key'"

# Try to render the test file
echo "🔨 Attempting to render test file..."
if quarto render test_invalid_key.qmd --to pdf 2>&1 | grep -q "CRITICAL ERROR"; then
    echo "✅ Error handling working correctly - build stopped as expected"
    echo "📋 Error output:"
    quarto render test_invalid_key.qmd --to pdf 2>&1 | grep -A 10 "CRITICAL ERROR"
else
    echo "❌ Error handling not working - build continued unexpectedly"
    quarto render test_invalid_key.qmd --to pdf 2>&1
fi

# Clean up
rm -f test_invalid_key.qmd test_invalid_key.pdf
echo "🧹 Cleaned up test files"
