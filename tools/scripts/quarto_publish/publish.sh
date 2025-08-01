#!/bin/bash

# MLSysBook Publish Script
# Uses binder for building and handles PDF deployment

set -e  # Exit on any error

echo "🚀 Starting MLSysBook publication process..."

# Check if we're on main branch
current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
    echo "❌ You are not on the main branch. Please switch to main and merge dev first."
    echo "   git checkout main"
    echo "   git merge dev"
    exit 1
fi

# Check if we have uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "❌ You have uncommitted changes. Please commit or stash them first."
    git status --short
    exit 1
fi

echo "✅ Git status check passed"

# Step 1: Clean any previous builds
echo "🧹 Cleaning previous builds..."
./binder clean
if [ $? -ne 0 ]; then
    echo "❌ Clean failed!"
    exit 1
fi

# Step 2: Build HTML version
echo "📚 Building HTML version..."
./binder build - html
if [ $? -ne 0 ]; then
    echo "❌ HTML build failed!"
    exit 1
else
    echo "✅ HTML build completed successfully."
fi

# Step 3: Build PDF version
echo "📄 Building PDF version..."
./binder build - pdf
if [ $? -ne 0 ]; then
    echo "❌ PDF build failed!"
    exit 1
else
    echo "✅ PDF build completed successfully."
fi

# Step 4: Check if PDF was created
PDF_PATH="build/pdf/Machine-Learning-Systems.pdf"
if [ ! -f "$PDF_PATH" ]; then
    echo "❌ PDF not found at $PDF_PATH"
    echo "📋 Checking build directory contents:"
    ls -la build/pdf/ || echo "build/pdf/ directory not found"
    exit 1
fi

echo "✅ PDF found at $PDF_PATH"

# Step 5: Copy PDF to assets directory
echo "📦 Copying PDF to assets directory..."
mkdir -p assets
cp "$PDF_PATH" assets/
if [ $? -ne 0 ]; then
    echo "❌ Failed to copy PDF to assets!"
    exit 1
else
    echo "✅ PDF copied to assets/Machine-Learning-Systems.pdf"
fi

# Step 6: Commit the PDF to assets
echo "💾 Committing PDF to assets..."
git add assets/Machine-Learning-Systems.pdf
git commit -m "📄 Add PDF to assets for download" || {
    echo "⚠️ No changes to commit (PDF might already be up to date)"
}

# Step 7: Push to main (this triggers GitHub Actions)
echo "🚀 Pushing to main branch..."
git push origin main
if [ $? -ne 0 ]; then
    echo "❌ Push to main failed!"
    exit 1
else
    echo "✅ Successfully pushed to main"
fi

echo ""
echo "🎉 Publication process completed!"
echo ""
echo "📊 What happened:"
echo "  ✅ Built HTML version using binder"
echo "  ✅ Built PDF version using binder"
echo "  ✅ Copied PDF to assets directory"
echo "  ✅ Committed PDF to git"
echo "  ✅ Pushed to main (triggers GitHub Actions)"
echo ""
echo "🌐 Your book will be available at:"
echo "  📖 Web version: https://harvard-edge.github.io/cs249r_book"
echo "  📄 PDF download: https://harvard-edge.github.io/cs249r_book/assets/Machine-Learning-Systems.pdf"
echo ""
echo "⏳ GitHub Actions will now:"
echo "  🔄 Run quality checks"
echo "  🏗️ Build all formats (Linux + Windows)"
echo "  🚀 Deploy to GitHub Pages"
echo "  📦 Create release assets"
echo ""
echo "📈 Monitor progress: https://github.com/harvard-edge/cs249r_book/actions"
