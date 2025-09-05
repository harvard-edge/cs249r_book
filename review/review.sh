#!/bin/bash
# Single command interface for textbook review system

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CHAPTER=""
AGENTS="all"
BRANCH_PREFIX="review"

# Function to show usage
usage() {
    echo "Usage: ./review.sh <chapter.qmd> [options]"
    echo ""
    echo "Options:"
    echo "  --agents <list>   Comma-separated list of agents (default: all)"
    echo "                    Available: junior_cs,senior_ee,masters,phd,industry"
    echo "  --branch <name>   Git branch prefix (default: review)"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./review.sh introduction.qmd"
    echo "  ./review.sh frameworks.qmd --agents junior_cs,masters,phd"
    echo "  ./review.sh efficient_ai.qmd --branch feature/improve"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --agents)
            AGENTS="$2"
            shift 2
            ;;
        --branch)
            BRANCH_PREFIX="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *.qmd)
            CHAPTER="$1"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Check if chapter specified
if [ -z "$CHAPTER" ]; then
    echo -e "${RED}Error: No chapter file specified${NC}"
    usage
    exit 1
fi

# Check if chapter exists
CHAPTER_PATH="quarto/contents/core/*/$CHAPTER"
if ! ls $CHAPTER_PATH 1> /dev/null 2>&1; then
    echo -e "${RED}Error: Chapter not found: $CHAPTER${NC}"
    exit 1
fi

# Get full path
FULL_PATH=$(ls $CHAPTER_PATH | head -1)
CHAPTER_NAME=$(basename "$CHAPTER" .qmd)

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Multi-Perspective Textbook Review System         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}📚 Chapter:${NC} $CHAPTER_NAME"
echo -e "${GREEN}📁 File:${NC} $FULL_PATH"
echo -e "${GREEN}👥 Agents:${NC} $AGENTS"
echo ""

# Create git branch
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BRANCH_NAME="$BRANCH_PREFIX-$CHAPTER_NAME-$TIMESTAMP"

echo -e "${BLUE}🔀 Creating git branch:${NC} $BRANCH_NAME"
git checkout -b "$BRANCH_NAME" 2>/dev/null || {
    echo -e "${RED}Warning: Could not create git branch${NC}"
}

# Run the review
echo -e "${BLUE}🚀 Starting review...${NC}"
echo ""

if [ "$AGENTS" == "all" ]; then
    python review/scripts/review_chapter.py "$FULL_PATH"
else
    python review/scripts/review_chapter.py "$FULL_PATH" --agents "$AGENTS"
fi

# Check if review was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Review completed successfully!${NC}"
    
    # Find the latest review results
    LATEST_REVIEW=$(ls -t review/temp/${CHAPTER_NAME}_*/review_results.json | head -1)
    
    if [ -n "$LATEST_REVIEW" ]; then
        echo -e "${GREEN}📊 Results saved to:${NC} $LATEST_REVIEW"
        
        # Show summary
        echo ""
        echo -e "${BLUE}Summary:${NC}"
        python -c "
import json
with open('$LATEST_REVIEW') as f:
    data = json.load(f)
    summary = data.get('summary', {})
    print(f'  Total Issues: {summary.get(\"total_issues\", 0)}')
    print(f'  Consensus Issues: {summary.get(\"consensus_issues\", 0)}')
    print(f'  Agents Used: {len(data.get(\"agents_used\", []))}')
"
    fi
else
    echo -e "${RED}❌ Review failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Review the findings in: $LATEST_REVIEW"
echo "  2. Apply improvements with: python apply_improvements.py $LATEST_REVIEW"
echo "  3. Commit changes: git add . && git commit -m 'feat: improve $CHAPTER_NAME'"
echo "