#!/bin/bash

# Repository Maintenance Script
# Run this periodically to keep the MLSysBook repository clean and healthy

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "🔧 MLSysBook Repository Maintenance"
echo "=================================="
echo "Repository: $REPO_ROOT"
echo "Date: $(date)"
echo ""

# Change to repository root
cd "$REPO_ROOT"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not a git repository"
    exit 1
fi

# Function to run health check
run_health_check() {
    echo "🏥 Running repository health check..."
    python3 tools/scripts/maintenance/repo_health_check.py --health-check
}

# Function to run full maintenance
run_full_maintenance() {
    echo "🔧 Running full maintenance..."
    python3 tools/scripts/maintenance/repo_health_check.py --full --remove-duplicates
}

# Function to run BFG cleanup (interactive)
run_bfg_cleanup() {
    echo "🗑️  Running BFG cleanup..."
    echo "⚠️  This will rewrite git history!"
    read -p "Continue with BFG cleanup? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 tools/scripts/maintenance/repo_health_check.py --bfg-cleanup
    else
        echo "BFG cleanup skipped"
    fi
}

# Function to show menu
show_menu() {
    echo ""
    echo "Maintenance Options:"
    echo "1) Health check only"
    echo "2) Full maintenance (clean artifacts, remove duplicates, optimize images)"
    echo "3) BFG cleanup (remove large files from history)"
    echo "4) All of the above"
    echo "5) Exit"
    echo ""
    read -p "Select option (1-5): " -n 1 -r
    echo
}

# Main execution
case "${1:-}" in
    "health")
        run_health_check
        ;;
    "full")
        run_full_maintenance
        ;;
    "bfg")
        run_bfg_cleanup
        ;;
    "all")
        run_health_check
        echo ""
        run_full_maintenance
        echo ""
        run_bfg_cleanup
        ;;
    *)
        # Interactive mode
        show_menu
        case $REPLY in
            1)
                run_health_check
                ;;
            2)
                run_full_maintenance
                ;;
            3)
                run_bfg_cleanup
                ;;
            4)
                run_health_check
                echo ""
                run_full_maintenance
                echo ""
                run_bfg_cleanup
                ;;
            5)
                echo "Exiting..."
                exit 0
                ;;
            *)
                echo "Invalid option"
                exit 1
                ;;
        esac
        ;;
esac

echo ""
echo "✅ Maintenance completed!"
echo "Repository: $REPO_ROOT"
echo "Date: $(date)" 