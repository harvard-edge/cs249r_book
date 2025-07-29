#!/bin/bash
#
# Format All Tables Script for MLSysBook
# 
# This script formats all grid tables in the book with proper alignment.
# Safe to run multiple times - uses intelligent analysis to avoid unnecessary changes.
#

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo -e "${BLUE}📚 MLSysBook Table Formatter${NC}"
echo -e "${BLUE}================================${NC}"

# Change to project root
cd "$PROJECT_ROOT"

# Check if book/contents exists
if [ ! -d "book/contents" ]; then
    echo -e "${RED}❌ Error: book/contents directory not found${NC}"
    echo -e "${YELLOW}💡 Make sure you're running this from the project root${NC}"
    exit 1
fi

# Check if the formatter script exists
FORMATTER_SCRIPT="tools/scripts/content/format_grid_tables.py"
if [ ! -f "$FORMATTER_SCRIPT" ]; then
    echo -e "${RED}❌ Error: Formatter script not found at $FORMATTER_SCRIPT${NC}"
    exit 1
fi

echo -e "${BLUE}🔍 Checking for tables that need formatting...${NC}"

# First, run in dry-run mode to see what would change
python3 "$FORMATTER_SCRIPT" --check-all --dry-run > /tmp/table_check.log 2>&1
check_exit_code=$?

if [ $check_exit_code -eq 0 ]; then
    echo -e "${GREEN}✨ All tables are already properly formatted!${NC}"
    exit 0
else
    echo -e "${YELLOW}📋 Found tables that need formatting${NC}"
    # Show the dry run results
    cat /tmp/table_check.log
    echo
    
    # Ask for confirmation unless --force is passed
    if [[ "$1" != "--force" ]]; then
        echo -e "${YELLOW}Do you want to format the tables? [y/N]${NC}"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}ℹ️  Table formatting cancelled${NC}"
            exit 0
        fi
    fi
    
    echo -e "${BLUE}🔧 Formatting tables...${NC}"
    
    # Run the formatter (without dry-run)
    if python3 "$FORMATTER_SCRIPT" --check-all --verbose; then
        echo
        echo -e "${GREEN}✅ Table formatting complete!${NC}"
        echo -e "${YELLOW}💡 Review the changes and commit when ready${NC}"
    else
        echo -e "${RED}❌ Table formatting failed${NC}"
        exit 1
    fi
fi 