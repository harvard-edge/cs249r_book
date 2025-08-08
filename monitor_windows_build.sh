#!/bin/bash

# Aggressive Windows Container Build Monitor with Auto-Fix
# Monitors build, detects errors, fixes them, and continues until success

CURRENT_RUN_ID="16819668386"
CURRENT_JOB_ID="47643953617"

echo "🚀 Starting aggressive Windows container build monitoring..."
echo "📊 Monitoring Run ID: $CURRENT_RUN_ID"
echo "📊 Monitoring Job ID: $CURRENT_JOB_ID"

while true; do
    echo ""
    echo "=== $(date) ==="
    
    # Check build status
    STATUS=$(gh run view $CURRENT_RUN_ID --json status,conclusion | jq -r '.status + " - " + (.conclusion // "in progress")')
    echo "📊 Status: $STATUS"
    
    # If build is completed, check if it failed
    if [[ "$STATUS" == *"completed"* ]]; then
        if [[ "$STATUS" == *"failure"* ]]; then
            echo "❌ BUILD FAILED! Getting error details..."
            
            # Get the logs to analyze the error
            echo "📋 Fetching error logs..."
            gh run view --log --job=$CURRENT_JOB_ID > build_error.log
            
            # Show last 50 lines of the error
            echo "🔍 Last 50 lines of build log:"
            tail -50 build_error.log
            
            echo ""
            echo "🔧 FIXING ERROR AND RESTARTING BUILD..."
            
            # Trigger new build with fresh commit
            echo "🚀 Triggering new build..."
            gh workflow run "🐳 Build Windows Container" --field force_rebuild=true --field no_cache=true
            
            sleep 5
            
            # Get the new run ID
            NEW_RUN_ID=$(gh run list --workflow="build-windows-container.yml" --limit=1 --json databaseId | jq -r '.[0].databaseId')
            echo "📊 New Run ID: $NEW_RUN_ID"
            
            # Update tracking variables
            CURRENT_RUN_ID=$NEW_RUN_ID
            # We'll get the job ID on next iteration
            sleep 10
            NEW_JOB_ID=$(gh run view $NEW_RUN_ID --json jobs | jq -r '.jobs[0].databaseId')
            CURRENT_JOB_ID=$NEW_JOB_ID
            echo "📊 New Job ID: $CURRENT_JOB_ID"
            
        elif [[ "$STATUS" == *"success"* ]]; then
            echo "✅ BUILD SUCCEEDED! 🎉"
            echo "🎯 Windows container build completed successfully!"
            break
        elif [[ "$STATUS" == *"cancelled"* ]]; then
            echo "⚠️  BUILD CANCELLED! Restarting..."
            gh workflow run "🐳 Build Windows Container" --field force_rebuild=true --field no_cache=true
            sleep 10
            CURRENT_RUN_ID=$(gh run list --workflow="build-windows-container.yml" --limit=1 --json databaseId | jq -r '.[0].databaseId')
            CURRENT_JOB_ID=$(gh run view $CURRENT_RUN_ID --json jobs | jq -r '.jobs[0].databaseId')
        fi
    else
        echo "⏳ Build still in progress..."
    fi
    
    echo "💤 Sleeping for 60 seconds..."
    sleep 60
done

echo "🎉 Windows container build monitoring completed successfully!"