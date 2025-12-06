#!/bin/bash
# Build Jupyter Book documentation
# Usage: ./scripts/build-book.sh

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📚 Building Jupyter Book documentation...${NC}"

# Check if jupyter-book is installed
if ! command -v jupyter-book &> /dev/null; then
    echo "❌ jupyter-book not found. Install it with:"
    echo "   pip install jupyter-book"
    exit 1
fi

# Build the book
echo -e "${BLUE}🔨 Running: jupyter-book build docs/${NC}"
jupyter-book build docs/

echo -e "${GREEN}✅ Book built successfully!${NC}"
echo -e "${BLUE}📖 Open: docs/_build/html/index.html${NC}"

# Optionally open in browser (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    read -p "Open in browser? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open docs/_build/html/index.html
    fi
fi
