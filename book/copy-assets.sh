#!/bin/bash
# Copy assets to build directory for Quarto

echo "📁 Copying assets to build directory..."

# Create assets directory if it doesn't exist
mkdir -p ../build/html/assets

# Copy assets directory
cp -r ../assets/* ../build/html/assets/

echo "✅ Assets copied successfully!" 