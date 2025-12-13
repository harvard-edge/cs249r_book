#!/bin/bash

# Allow bypassing the check with an environment variable
if [ "$ALLOW_WORKFLOW_CHANGES" = "true" ]; then
    echo "✅ Bypassing workflow change check due to ALLOW_WORKFLOW_CHANGES flag"
    exit 0
fi

# Check if any workflow files are being committed
if git diff --cached --name-only | grep -q "\.github/workflows/"; then
    echo "⚠️  Workflow files detected in commit!"
    echo "📋 This may cause issues with publish-live workflow."
    echo "💡 Consider:"
    echo "   1. Manual merge workflow changes first"
    echo "   2. Then run publish-live for content only"
    echo "   3. Or use a PAT with workflows:write permission"
    echo ""
    echo "🔍 Workflow files being committed:"
    git diff --cached --name-only | grep "\.github/workflows/"
    echo ""
    read -p "Continue with commit? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Commit cancelled"
        exit 1
    fi
    echo "✅ Proceeding with workflow file changes"
fi
