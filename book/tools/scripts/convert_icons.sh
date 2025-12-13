#!/bin/bash

# Convert foldbox icons from PNG to PDF for LaTeX rendering
# This script converts all the required icon files for the custom-numbered-blocks extension

set -e  # Exit on any error

echo "🔄 Converting foldbox icons from PNG to PDF..."

# Change to the icons directory
cd book/_extensions/ute/custom-numbered-blocks/style/icons

# Check if ImageMagick is installed
if ! command -v convert &> /dev/null; then
    echo "❌ Error: ImageMagick is not installed."
    echo "Please install it first:"
    echo "  macOS: brew install imagemagick"
    echo "  Ubuntu: sudo apt-get install imagemagick"
    exit 1
fi

# Convert all required icons
echo "📁 Converting icons..."

convert icon_callout-quiz-question.png icon_callout-quiz-question.pdf
convert icon_callout-quiz-answer.png icon_callout-quiz-answer.pdf
convert icon_callout-chapter-connection.png icon_callout-chapter-connection.pdf
convert icon_callout-resource-exercises.png icon_callout-resource-exercises.pdf
convert Icon_callout-resource-slides.png icon_callout-resource-slides.pdf
convert Icon_callout-resource-videos.png icon_callout-resource-videos.pdf

echo "✅ All icons converted successfully!"
echo "📋 Converted files:"
ls -la icon_*.pdf

echo ""
echo "🎉 Icon conversion complete. PDF build should now work without icon errors."
