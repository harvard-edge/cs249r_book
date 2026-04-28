#!/bin/bash

# Ensure we're in the right directory
cd /Users/VJ/GitHub/MLSysBook-yaml-audit

echo "Waiting for the initial full pass (Pass 0) to complete..."
while kill -0 26149 2>/dev/null; do
    sleep 5
done
echo "Pass 0 completed."

git add .
git commit -m "Auto-audit Pass 0 (All files)" || true

for pass in {1..10}; do
    echo "==================================="
    echo "Starting Iteration $pass..."
    echo "==================================="
    
    # Get YAML files modified in the previous pass
    FILES=$(git diff --name-only HEAD~1 | grep '\.yaml$' || true)
    
    if [ -z "$FILES" ]; then
        echo "No files were modified in the previous pass. Convergence reached!"
        break
    fi
    
    FILE_COUNT=$(echo "$FILES" | wc -l | tr -d ' ')
    echo "Found $FILE_COUNT updated YAML files to review."
    
    # Write the files to a temporary list
    echo "$FILES" > .files_to_audit.txt
    
    # Run the python script with this specific list of files
    python3 ../MLSysBook/audit_yamls_subset.py .files_to_audit.txt >> iterative_audit.log 2>&1
    
    git add .
    # If there are changes, commit them
    if git diff --cached --quiet; then
        echo "Pass $pass made no changes. Convergence reached!"
        break
    else
        git commit -m "Auto-audit Pass $pass"
    fi
done

echo "Iterative audit process finished!"