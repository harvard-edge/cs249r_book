#!/bin/bash
# Re-download images with better filenames

echo "🔄 Re-downloading images with proper filenames"
echo "=============================================="

# Remove old auto-* images
echo "🗑️  Removing old auto-* images..."
find quarto/contents/labs -type f -name "auto-*" -delete

# Also remove the directories if they're empty
find quarto/contents/labs -type d -name "images" -empty -delete
find quarto/contents/labs -type d \( -name "png" -o -name "jpg" -o -name "jpeg" \) -empty -delete

echo "✅ Old images removed"
echo ""

# Re-download all images from labs
echo "📥 Re-downloading images with new naming..."
python3 tools/scripts/manage_external_images.py -d quarto/contents/labs/

echo ""
echo "✅ Done! Images have been re-downloaded with proper filenames"
