#!/bin/bash
# Repository History Cleanup Script
# Removes large files from Git history using BFG Repo-Cleaner
#
# WARNING: This rewrites Git history. Make sure you have a backup!

set -e  # Exit on error

REPO_DIR="/Users/VJ/GitHub/TinyTorch"
BACKUP_DIR="${REPO_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
CLEAN_REPO_DIR="${REPO_DIR}_clean"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== TinyTorch Repository History Cleanup ===${NC}\n"

# Check if BFG is installed
if ! command -v bfg &> /dev/null; then
    echo -e "${RED}ERROR: BFG Repo-Cleaner is not installed.${NC}"
    echo "Install with: brew install bfg"
    echo "Or download from: https://rtyley.github.io/bfg-repo-cleaner/"
    exit 1
fi

# Check if we're in the right directory
if [ ! -d "$REPO_DIR/.git" ]; then
    echo -e "${RED}ERROR: Not a Git repository: $REPO_DIR${NC}"
    exit 1
fi

# Safety check: warn about uncommitted changes
if [ -n "$(git -C "$REPO_DIR" status --porcelain)" ]; then
    echo -e "${YELLOW}WARNING: You have uncommitted changes!${NC}"
    echo "Please commit or stash them before proceeding."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create backup
echo -e "${GREEN}Step 1: Creating backup...${NC}"
git -C "$REPO_DIR" clone --mirror "$REPO_DIR" "$BACKUP_DIR"
echo -e "${GREEN}✓ Backup created: $BACKUP_DIR${NC}\n"

# Create mirror clone for BFG
echo -e "${GREEN}Step 2: Creating mirror clone for BFG...${NC}"
rm -rf "$CLEAN_REPO_DIR"
git clone --mirror "$REPO_DIR" "$CLEAN_REPO_DIR"
echo -e "${GREEN}✓ Mirror clone created${NC}\n"

# Change to clean repo directory
cd "$CLEAN_REPO_DIR"

# Remove large files/folders
echo -e "${GREEN}Step 3: Removing large files from history...${NC}"

# Remove CIFAR-10 dataset files
echo "  - Removing CIFAR-10 dataset files..."
bfg --delete-folders cifar-10-batches-py 2>&1 | grep -v "^Using.*repo" || true

# Remove virtual environment directories
echo "  - Removing virtual environment directories..."
bfg --delete-folders bin 2>&1 | grep -v "^Using.*repo" || true
bfg --delete-folders lib 2>&1 | grep -v "^Using.*repo" || true
bfg --delete-folders include 2>&1 | grep -v "^Using.*repo" || true
bfg --delete-folders share 2>&1 | grep -v "^Using.*repo" || true

# Remove large GIF files (optional - comment out if you want to keep them)
echo "  - Removing large GIF files..."
bfg --delete-files "*.gif" --no-blob-protection 2>&1 | grep -v "^Using.*repo" || true

# Remove large PNG files (optional - comment out if you want to keep them)
echo "  - Removing large PNG files..."
bfg --delete-files "Gemini_Generated_Image_*.png" --no-blob-protection 2>&1 | grep -v "^Using.*repo" || true

# Remove pyvenv.cfg
echo "  - Removing pyvenv.cfg..."
bfg --delete-files pyvenv.cfg 2>&1 | grep -v "^Using.*repo" || true

echo -e "${GREEN}✓ Files removed${NC}\n"

# Clean up Git
echo -e "${GREEN}Step 4: Cleaning up Git repository...${NC}"
git reflog expire --expire=now --all
git gc --prune=now --aggressive
echo -e "${GREEN}✓ Cleanup complete${NC}\n"

# Show results
echo -e "${GREEN}Step 5: Results${NC}"
CLEAN_SIZE=$(du -sh . | cut -f1)
echo "  Clean repository size: $CLEAN_SIZE"

echo -e "\n${YELLOW}=== Next Steps ===${NC}"
echo "1. Review the cleaned repository:"
echo "   cd $CLEAN_REPO_DIR"
echo "   git log --oneline -10"
echo ""
echo "2. If satisfied, replace original .git:"
echo "   cd $REPO_DIR"
echo "   mv .git .git.backup"
echo "   cp -r $CLEAN_REPO_DIR $REPO_DIR/.git"
echo ""
echo "3. Verify:"
echo "   cd $REPO_DIR"
echo "   git status"
echo ""
echo "4. Force push to GitHub (WARNING: rewrites history):"
echo "   git push origin --force --all"
echo "   git push origin --force --tags"
echo ""
echo -e "${YELLOW}Backup location: $BACKUP_DIR${NC}"
echo -e "${YELLOW}Clean repo location: $CLEAN_REPO_DIR${NC}"
