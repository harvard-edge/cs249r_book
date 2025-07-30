#!/bin/bash

# Check if the current git branch is main
current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
    echo "You are not on the main branch. Please switch to the main branch to proceed. You should have merged dev into main by now."
    exit 1
fi

# Step 1: Run quarto render
echo "Running quarto render..."
cd book
quarto render --to html
if [ $? -ne 0 ]; then
    echo "quarto render failed!"
    exit 1
else
    echo "quarto render completed successfully."
fi

# Step 2: Run the Python script to compress the PDF (if it exists)
echo "Compressing PDF..."
if [ -f "./_book/Machine-Learning-Systems.pdf" ]; then
    python3 ../.github/scripts/gs_compress_pdf.py -i ./_book/Machine-Learning-Systems.pdf -o ./_book/ebook.pdf -s "/ebook"
    if [ $? -ne 0 ]; then
        echo "PDF compression failed!"
        exit 1
    else
        echo "PDF compression completed successfully."
        # Replace original with compressed
        mv ./_book/ebook.pdf ./_book/Machine-Learning-Systems.pdf
        echo "PDF replaced successfully."
    fi
else
    echo "No PDF found to compress, skipping PDF compression step."
fi

# Step 3: Commit and push the built files to main branch
echo "Committing built files to main branch..."
cd ..  # Back to root directory
git add book/_book/
git commit -m "Build: Update site $(date '+%Y-%m-%d %H:%M:%S')" || echo "No changes to commit"
git push origin main

# Netlify will automatically deploy from main branch
echo "✅ Files pushed to main branch. Netlify will automatically deploy the changes."
echo "🌐 Your site will be available at your Netlify URL shortly."

echo "All steps completed successfully."
