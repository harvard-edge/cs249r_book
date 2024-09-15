#!/bin/bash

# Check if the current git branch is main
current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
    echo "You are not on the main branch. Please switch to the main branch to proceed. You should have merged dev into main by now."
    exit 1
fi

# Step 1: Run quarto render
echo "Running quarto render..."
quarto render
if [ $? -ne 0 ]; then
    echo "quarto render failed!"
    exit 1
else
    echo "quarto render completed successfully."
fi

# Step 2: Run the Python script to compress the PDF
echo "Compressing PDF..."
python3 ./scripts/quarto_publish/gs_compress_pdf.py -i ./_book/Machine-Learning-Systems.pdf -o ./_book/ebook.pdf -s "/ebook"
if [ $? -ne 0 ]; then
    echo "PDF compression failed!"
    exit 1
else    
    echo "PDF compression completed successfully."
fi

# Step 3: Move the compressed PDF to the original file name
echo "Replacing the original PDF with the compressed PDF..."
mv ./_book/ebook.pdf ./_book/Machine-Learning-Systems.pdf
if [ $? -ne 0 ]; then
    echo "Failed to replace the original PDF!"
    exit 1
else    
    echo "PDF replaced successfully."
fi

# Step 4: Publish to gh-pages without rendering
echo "Publishing to gh-pages..."
quarto publish --no-render gh-pages
if [ $? -ne 0 ]; then
    echo "Publishing to gh-pages failed!"
    exit 1
else    
    echo "Published to gh-pages successfully."
fi

echo "All steps completed successfully."

