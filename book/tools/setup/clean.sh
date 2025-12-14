#!/bin/bash

# Git Repository Maintenance Scripts
# Run these regularly to keep your repository healthy

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# ==========================================
# WEEKLY MAINTENANCE
# ==========================================

weekly_maintenance() {
    print_header "WEEKLY MAINTENANCE"

    # 1. Clean up merged branches
    print_info "Analyzing merged branches..."

    # Get current branch
    current_branch=$(git branch --show-current)
    print_info "Current branch: $current_branch"

    # Find merged branches (excluding main/master/dev)
    merged_branches=$(git branch --merged | grep -v -E "(main|master|dev|\*)" | sed 's/^[[:space:]]*//')

    if [ -n "$merged_branches" ]; then
        echo -e "\n${YELLOW}Merged branches found:${NC}"
        echo "$merged_branches" | while read branch; do
            if [ -n "$branch" ]; then
                last_commit=$(git log -1 --pretty=format:'%cr (%h) %s' "$branch" 2>/dev/null)
                echo "  📦 $branch - Last commit: $last_commit"
            fi
        done

        echo -e "\n${YELLOW}These branches have been merged into the current branch and can be safely deleted.${NC}"
        echo "They will be removed from your local repository only (not remote)."
        read -p "Delete these merged branches? (y/N): " confirm

        if [[ $confirm =~ ^[Yy]$ ]]; then
            echo "$merged_branches" | while read branch; do
                if [ -n "$branch" ]; then
                    echo "  Deleting: $branch"
                    git branch -d "$branch"
                fi
            done
            print_success "Local merged branches deleted"
        else
            print_info "Skipped branch deletion"
        fi
    else
        print_success "No merged branches to clean up"
    fi

    # 2. Clean up remote tracking branches
    print_info "Analyzing stale remote references..."

    # Show what will be pruned
    stale_refs=$(git remote prune origin --dry-run 2>/dev/null | grep "would prune" || true)

    if [ -n "$stale_refs" ]; then
        echo -e "\n${YELLOW}Stale remote references found:${NC}"
        echo "$stale_refs" | sed 's/^[[:space:]]*\*[[:space:]]*would prune[[:space:]]*/  🗑️  /'
        echo -e "\n${YELLOW}These are remote branches that no longer exist on the remote repository.${NC}"
        echo "Removing these references will clean up your 'git branch -r' output."
        read -p "Remove stale remote references? (y/N): " confirm

        if [[ $confirm =~ ^[Yy]$ ]]; then
            git remote prune origin
            print_success "Stale remote references cleaned"
        else
            print_info "Skipped remote reference cleanup"
        fi
    else
        print_success "No stale remote references found"
    fi

    # 3. Clean up reflog
    print_info "Analyzing reflog entries..."

    # Count current reflog entries
    current_reflog_count=$(git reflog --all | wc -l | tr -d ' ')
    old_reflog_count=$(git reflog --all --until="30 days ago" | wc -l | tr -d ' ')

    if [ "$old_reflog_count" -gt 0 ]; then
        echo -e "\n${YELLOW}Reflog cleanup details:${NC}"
        echo "  📊 Total reflog entries: $current_reflog_count"
        echo "  🗑️  Entries older than 30 days: $old_reflog_count"
        echo "  ✅ Entries to keep: $((current_reflog_count - old_reflog_count))"
        echo -e "\n${YELLOW}The reflog tracks where your HEAD has been - old entries are safe to remove.${NC}"
        echo "This helps keep your repository's internal storage optimized."
        read -p "Remove reflog entries older than 30 days? (y/N): " confirm

        if [[ $confirm =~ ^[Yy]$ ]]; then
            git reflog expire --expire=30.days --expire-unreachable=7.days --all
            new_count=$(git reflog --all | wc -l | tr -d ' ')
            print_success "Reflog cleaned - removed $((current_reflog_count - new_count)) entries"
        else
            print_info "Skipped reflog cleanup"
        fi
    else
        print_success "No old reflog entries to clean"
    fi

    # 4. Basic garbage collection
    print_info "Analyzing repository for garbage collection..."

    # Get current size
    current_size=$(du -sh .git/ | awk '{print $1}')

    # Check if GC is needed
    loose_objects=$(find .git/objects -type f | wc -l | tr -d ' ')
    packs=$(find .git/objects/pack -name "*.pack" | wc -l | tr -d ' ')

    echo -e "\n${YELLOW}Garbage collection analysis:${NC}"
    echo "  📁 Current repository size: $current_size"
    echo "  📦 Loose objects: $loose_objects"
    echo "  🗜️  Pack files: $packs"

    if [ "$loose_objects" -gt 100 ] || [ "$packs" -gt 10 ]; then
        echo -e "\n${YELLOW}Garbage collection recommended to optimize storage.${NC}"
        echo "This will pack loose objects and optimize the repository structure."
        read -p "Run garbage collection? (y/N): " confirm

        if [[ $confirm =~ ^[Yy]$ ]]; then
            git gc --auto
            new_size=$(du -sh .git/ | awk '{print $1}')
            print_success "Garbage collection completed - size: $current_size → $new_size"
        else
            print_info "Skipped garbage collection"
        fi
    else
        print_success "Repository is already well optimized"
    fi
}

# ==========================================
# MONTHLY MAINTENANCE
# ==========================================

monthly_maintenance() {
    print_header "MONTHLY MAINTENANCE"

    # 1. Repository size check
    print_info "Checking repository size..."
    repo_size=$(du -sh .git/ | awk '{print $1}')
    echo "Repository size: $repo_size"

    # 2. Find large files
    print_info "Analyzing large files in history..."
    echo "Top 10 largest files:"
    git rev-list --objects --all | \
    git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
    awk '/^blob/ {print substr($0,6)}' | \
    sort --numeric-sort --key=2 --reverse | \
    head -10 | \
    while read line; do
        size=$(echo $line | awk '{print $2}')
        filename=$(echo $line | awk '{print $3}')
        if command -v bc &> /dev/null; then
            size_mb=$(echo "scale=1; $size / 1024 / 1024" | bc)
            echo "  ${size_mb}MB - $filename"
        else
            echo "  ${size} bytes - $filename"
        fi
    done

    # 3. Check for common temporary files
    print_info "Checking for temporary files that shouldn't be tracked..."
    temp_files=$(git ls-files | grep -E '\.(tmp|log|cache|DS_Store)$|__pycache__|\.pyc$' || true)
    if [ -n "$temp_files" ]; then
        print_warning "Found temporary files in repository:"
        echo "$temp_files"
        echo "Consider adding these patterns to .gitignore"
    else
        print_success "No obvious temporary files found"
    fi

    # 4. Aggressive garbage collection
    print_info "Analyzing for aggressive garbage collection..."

    current_size=$(du -sh .git/ | awk '{print $1}')

    # Check for conditions that benefit from aggressive GC
    loose_objects=$(find .git/objects -type f | wc -l | tr -d ' ')
    old_packs=$(find .git/objects/pack -name "*.pack" -mtime +30 | wc -l | tr -d ' ')

    echo -e "\n${YELLOW}Aggressive cleanup analysis:${NC}"
    echo "  📁 Current size: $current_size"
    echo "  📦 Loose objects: $loose_objects"
    echo "  🗓️  Old pack files (30+ days): $old_packs"
    echo -e "\n${YELLOW}Aggressive GC will deeply optimize the repository but takes longer.${NC}"
    echo "It repacks all objects and optimizes delta compression."
    read -p "Run aggressive garbage collection? (y/N): " confirm

    if [[ $confirm =~ ^[Yy]$ ]]; then
        print_info "Running aggressive cleanup (this may take a few minutes)..."
        git gc --aggressive --prune=30.days.ago
        new_size=$(du -sh .git/ | awk '{print $1}')
        print_success "Aggressive cleanup completed - size: $current_size → $new_size"
    else
        print_info "Skipped aggressive cleanup"
    fi

    # 5. Check remote branches that might need cleanup
    print_info "Analyzing remote branches for potential cleanup..."

    main_branch=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo "main")
    remote_merged=$(git branch -r --merged "origin/$main_branch" 2>/dev/null | grep -v -E "(origin/$main_branch|origin/master|origin/dev|HEAD)" | sed 's/^[[:space:]]*//' || true)

    if [ -n "$remote_merged" ]; then
        echo -e "\n${YELLOW}Remote branches merged into $main_branch:${NC}"
        echo "$remote_merged" | while read branch; do
            if [ -n "$branch" ]; then
                last_commit=$(git log -1 --pretty=format:'%cr (%h) %s' "$branch" 2>/dev/null || echo "unknown")
                echo "  🌿 $branch - Last commit: $last_commit"
            fi
        done
        echo -e "\n${YELLOW}These remote branches appear to be merged.${NC}"
        echo "You may want to delete them from the remote repository."
        echo "Note: This analysis is for information only - no automatic deletion of remote branches."
        print_warning "To delete remote branches, use: git push origin --delete <branch-name>"
    else
        print_success "No obviously merged remote branches found"
    fi
}

# ==========================================
# QUARTERLY MAINTENANCE
# ==========================================

quarterly_maintenance() {
    print_header "QUARTERLY MAINTENANCE (Deep Clean)"

    # 1. Full repository analysis
    print_info "Running full repository analysis..."
    if command -v git-sizer &> /dev/null; then
        git-sizer --verbose
    else
        print_warning "git-sizer not installed. Install with: brew install git-sizer"
        echo "Repository statistics:"
        echo "  Commits: $(git rev-list --all --count)"
        echo "  Branches: $(git branch -a | wc -l)"
        echo "  Files: $(git ls-files | wc -l)"
    fi

    # 2. Check for large directories
    print_info "Checking directory sizes in working tree..."
    find . -type d -not -path './.git/*' -not -path './node_modules/*' | \
    while read dir; do
        size=$(du -sh "$dir" 2>/dev/null | awk '{print $1}')
        echo "$size - $dir"
    done | sort -hr | head -10

    # 3. Identify old branches
    print_info "Identifying old branches (not updated in 90 days)..."
    git for-each-ref --format='%(refname:short) %(committerdate:relative)' refs/heads/ | \
    grep -E '(months|year)' | head -10

    # 4. Check for duplicate files
    print_info "Checking for potential duplicate large files..."
    git ls-files | xargs -I {} sh -c 'echo "$(git log -1 --pretty=format:"%H" -- "{}" 2>/dev/null || echo "unknown") $(du -h "{}" 2>/dev/null | cut -f1) {}"' | \
    sort -k2 -hr | head -10
}

# ==========================================
# EMERGENCY CLEANUP
# ==========================================

emergency_cleanup() {
    print_header "EMERGENCY CLEANUP"

    current_size=$(du -sh .git/ | awk '{print $1}')
    reflog_entries=$(git reflog --all | wc -l | tr -d ' ')
    loose_objects=$(find .git/objects -type f | wc -l | tr -d ' ')

    echo -e "${RED}⚠️  EMERGENCY CLEANUP - AGGRESSIVE OPERATIONS ⚠️${NC}"
    echo -e "\n${YELLOW}Current repository status:${NC}"
    echo "  📁 Repository size: $current_size"
    echo "  📝 Reflog entries: $reflog_entries"
    echo "  📦 Loose objects: $loose_objects"

    echo -e "\n${RED}This emergency cleanup will:${NC}"
    echo "  🗑️  Remove ALL reflog entries (lose history of where HEAD was)"
    echo "  🧹 Run aggressive garbage collection"
    echo "  📦 Repack all objects for maximum compression"
    echo "  ⚠️  This is IRREVERSIBLE but generally safe"

    echo -e "\n${YELLOW}Use this when:${NC}"
    echo "  - Repository is extremely bloated"
    echo "  - Normal cleanup isn't sufficient"
    echo "  - You don't need reflog history"

    read -p "Continue with emergency cleanup? This cannot be undone! (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        print_info "Emergency cleanup cancelled"
        return
    fi

    # Final confirmation
    read -p "Are you absolutely sure? Type 'YES' to continue: " final_confirm
    if [[ "$final_confirm" != "YES" ]]; then
        print_info "Emergency cleanup cancelled"
        return
    fi

    # 1. Remove all reflog entries
    print_info "Clearing all reflog entries..."
    git reflog expire --expire=now --all

    # 2. Aggressive garbage collection
    print_info "Running aggressive garbage collection..."
    git gc --aggressive --prune=now

    # 3. Repack repository
    print_info "Repacking repository..."
    git repack -ad

    # 4. Show results
    new_size=$(du -sh .git/ | awk '{print $1}')
    print_success "Emergency cleanup completed. New size: $new_size"
}

# ==========================================
# UTILITIES
# ==========================================

show_repo_stats() {
    print_header "REPOSITORY STATISTICS"

    echo "Repository size: $(du -sh .git/ | awk '{print $1}')"
    echo "Working directory size: $(du -sh . --exclude=.git | awk '{print $1}')"
    echo "Total commits: $(git rev-list --all --count)"
    echo "Total branches: $(git branch -a | wc -l)"
    echo "Total files tracked: $(git ls-files | wc -l)"
    echo "Last commit: $(git log -1 --pretty=format:'%cr (%h)')"

    # Show branch activity
    echo -e "\nRecent branch activity:"
    git for-each-ref --sort=-committerdate --format='%(refname:short) - %(committerdate:relative)' refs/heads/ | head -5
}

check_gitignore() {
    print_header "GITIGNORE RECOMMENDATIONS"

    # Common patterns that should be ignored
    patterns_to_check=(
        "*.tmp"
        "*.log"
        "*.cache"
        ".DS_Store"
        "__pycache__"
        "*.pyc"
        "node_modules"
        ".env"
        "*.safetensors"
        "*.bin"
        "*.h5"
        "*.pkl"
    )

    for pattern in "${patterns_to_check[@]}"; do
        if git ls-files | grep -E "${pattern//\*/.*}" &>/dev/null; then
            if ! grep -q "$pattern" .gitignore 2>/dev/null; then
                print_warning "Consider adding '$pattern' to .gitignore"
            fi
        fi
    done
}

# ==========================================
# MAIN MENU
# ==========================================

show_menu() {
    echo -e "\n${BLUE}Git Repository Maintenance Menu${NC}"
    echo "================================="
    echo "1. Weekly Maintenance (5 mins)"
    echo "2. Monthly Maintenance (10 mins)"
    echo "3. Quarterly Maintenance (15 mins)"
    echo "4. Emergency Cleanup (USE WITH CAUTION)"
    echo "5. Show Repository Statistics"
    echo "6. Check .gitignore Recommendations"
    echo "7. Run All Regular Maintenance"
    echo "8. Exit"
    echo
}

main() {
    if [ ! -d ".git" ]; then
        print_warning "Not in a git repository!"
        exit 1
    fi

    while true; do
        show_menu
        read -p "Select option (1-8): " choice

        case $choice in
            1) weekly_maintenance ;;
            2) monthly_maintenance ;;
            3) quarterly_maintenance ;;
            4) emergency_cleanup ;;
            5) show_repo_stats ;;
            6) check_gitignore ;;
            7)
                weekly_maintenance
                monthly_maintenance
                quarterly_maintenance
                ;;
            8)
                print_success "Goodbye!"
                exit 0
                ;;
            *)
                print_warning "Invalid option. Please select 1-8."
                ;;
        esac

        echo
        read -p "Press Enter to continue..."
    done
}

# If script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

# ==========================================
# INDIVIDUAL FUNCTIONS FOR COMMAND LINE USE
# ==========================================

# You can also call individual functions:
# ./maintenance.sh weekly_maintenance
# ./maintenance.sh monthly_maintenance
# ./maintenance.sh show_repo_stats

if [[ $# -gt 0 ]] && [[ "$1" != "main" ]]; then
    if declare -f "$1" > /dev/null; then
        "$@"
    else
        echo "Function '$1' not found"
        echo "Available functions: weekly_maintenance, monthly_maintenance, quarterly_maintenance, show_repo_stats, check_gitignore"
    fi
fi
