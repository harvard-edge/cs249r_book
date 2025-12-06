#!/bin/bash
# Build website documentation (static site)
# Usage: ./scripts/build-docs.sh

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🌐 Building TinyTorch website documentation...${NC}"

# Check if docs/_build exists
if [ ! -d "docs/_build" ]; then
    echo -e "${YELLOW}⚠️  No book build found. Building Jupyter Book first...${NC}"
    ./scripts/build-book.sh
fi

# Generate static assets
echo -e "${BLUE}📦 Generating static assets...${NC}"

# Copy static files to website directory (customize as needed)
if [ -d "website" ]; then
    echo -e "${BLUE}📋 Copying book output to website/...${NC}"
    mkdir -p website/docs
    cp -r docs/_build/html/* website/docs/
    echo -e "${GREEN}✅ Documentation copied to website/docs/${NC}"
else
    echo -e "${YELLOW}ℹ️  No website/ directory found.${NC}"
    echo -e "${YELLOW}   Book output is in: docs/_build/html/${NC}"
fi

echo -e "${GREEN}✅ Documentation build complete!${NC}"
echo -e "${BLUE}📖 View at: docs/_build/html/index.html${NC}"
