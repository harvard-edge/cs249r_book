#!/bin/bash
# Merge backup site/ into docs/ while preserving updated documentation
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔄 Merging site/ backup into docs/"
echo "=================================="
echo ""

# Find the backup directory
BACKUP_DIR=$(ls -dt ../TinyTorch-backup-* 2>/dev/null | head -1)

if [ -z "$BACKUP_DIR" ]; then
    echo "❌ No backup directory found!"
    echo "   Expected: ../TinyTorch-backup-*"
    exit 1
fi

echo "📦 Found backup: $BACKUP_DIR"
echo ""

if [ ! -d "$BACKUP_DIR/site" ]; then
    echo "❌ Backup site/ directory not found!"
    exit 1
fi

echo "📋 Copying website files from backup..."

# Copy website build files
if [ -f "$BACKUP_DIR/site/build.sh" ]; then
    cp "$BACKUP_DIR/site/build.sh" docs/
    chmod +x docs/build.sh
    echo "  ✅ build.sh"
fi

if [ -f "$BACKUP_DIR/site/_config.yml" ]; then
    cp "$BACKUP_DIR/site/_config.yml" docs/
    echo "  ✅ _config.yml"
fi

if [ -f "$BACKUP_DIR/site/_toc.yml" ]; then
    cp "$BACKUP_DIR/site/_toc.yml" docs/
    echo "  ✅ _toc.yml"
fi

if [ -f "$BACKUP_DIR/site/conf.py" ]; then
    cp "$BACKUP_DIR/site/conf.py" docs/
    echo "  ✅ conf.py"
fi

if [ -f "$BACKUP_DIR/site/Makefile" ]; then
    cp "$BACKUP_DIR/site/Makefile" docs/
    echo "  ✅ Makefile"
fi

if [ -f "$BACKUP_DIR/site/requirements.txt" ]; then
    cp "$BACKUP_DIR/site/requirements.txt" docs/
    echo "  ✅ requirements.txt"
fi

echo ""
echo "📁 Copying website content directories..."

# Copy website content directories
if [ -d "$BACKUP_DIR/site/modules" ]; then
    cp -r "$BACKUP_DIR/site/modules" docs/
    echo "  ✅ modules/"
fi

if [ -d "$BACKUP_DIR/site/chapters" ]; then
    cp -r "$BACKUP_DIR/site/chapters" docs/
    echo "  ✅ chapters/"
fi

if [ -d "$BACKUP_DIR/site/tito" ]; then
    cp -r "$BACKUP_DIR/site/tito" docs/
    echo "  ✅ tito/"
fi

if [ -d "$BACKUP_DIR/site/tiers" ]; then
    cp -r "$BACKUP_DIR/site/tiers" docs/
    echo "  ✅ tiers/"
fi

if [ -d "$BACKUP_DIR/site/usage-paths" ]; then
    cp -r "$BACKUP_DIR/site/usage-paths" docs/
    echo "  ✅ usage-paths/"
fi

echo ""
echo "📝 Copying website markdown files..."

# Copy top-level markdown files (website content)
WEBSITE_MD_FILES=(
    "intro.md"
    "getting-started.md"
    "quickstart-guide.md"
    "student-workflow.md"
    "learning-progress.md"
    "learning-journey-visual.md"
    "checkpoint-system.md"
    "community.md"
    "datasets.md"
    "faq.md"
    "for-instructors.md"
    "instructor-guide.md"
    "prerequisites.md"
    "resources.md"
    "credits.md"
)

for md_file in "${WEBSITE_MD_FILES[@]}"; do
    if [ -f "$BACKUP_DIR/site/$md_file" ]; then
        cp "$BACKUP_DIR/site/$md_file" docs/
        echo "  ✅ $md_file"
    fi
done

echo ""
echo "📄 Copying additional site files..."

# Copy other site-specific files
if [ -f "$BACKUP_DIR/site/prepare_notebooks.sh" ]; then
    cp "$BACKUP_DIR/site/prepare_notebooks.sh" docs/
    chmod +x docs/prepare_notebooks.sh
    echo "  ✅ prepare_notebooks.sh"
fi

if [ -f "$BACKUP_DIR/site/build_pdf.sh" ]; then
    cp "$BACKUP_DIR/site/build_pdf.sh" docs/
    chmod +x docs/build_pdf.sh
    echo "  ✅ build_pdf.sh"
fi

if [ -f "$BACKUP_DIR/site/build_pdf_simple.sh" ]; then
    cp "$BACKUP_DIR/site/build_pdf_simple.sh" docs/
    chmod +x docs/build_pdf_simple.sh
    echo "  ✅ build_pdf_simple.sh"
fi

if [ -f "$BACKUP_DIR/site/references.bib" ]; then
    cp "$BACKUP_DIR/site/references.bib" docs/
    echo "  ✅ references.bib"
fi

if [ -f "$BACKUP_DIR/site/README.md" ]; then
    cp "$BACKUP_DIR/site/README.md" docs/website-README.md
    echo "  ✅ README.md (as website-README.md)"
fi

if [ -f "$BACKUP_DIR/site/NAVIGATION_REDESIGN_SUMMARY.md" ]; then
    cp "$BACKUP_DIR/site/NAVIGATION_REDESIGN_SUMMARY.md" docs/
    echo "  ✅ NAVIGATION_REDESIGN_SUMMARY.md"
fi

echo ""
echo "🖼️  Copying _static directory (preserving demos/)..."

# Copy _static but preserve our updated demos/
if [ -d "$BACKUP_DIR/site/_static" ]; then
    # Copy everything except demos
    for item in "$BACKUP_DIR/site/_static"/*; do
        basename_item=$(basename "$item")
        if [ "$basename_item" != "demos" ]; then
            cp -r "$item" docs/_static/
            echo "  ✅ _static/$basename_item"
        fi
    done
fi

echo ""
echo "✅ Merge Complete!"
echo "=================="
echo ""
echo "📁 docs/ now contains:"
echo "  ✅ Jupyter Book website files (from backup)"
echo "  ✅ Updated docs/development/ (preserved)"
echo "  ✅ Updated docs/instructor/ (preserved)"
echo "  ✅ Updated docs/_static/demos/ (preserved)"
echo ""
echo "🔍 Next: Verify website builds"
echo "   cd docs && ./build.sh"
echo ""
