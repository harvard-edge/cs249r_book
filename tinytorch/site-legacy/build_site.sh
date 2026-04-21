#!/bin/bash

# 1. Activate the virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv directory not found."
    exit 1
fi

# 2. Clear the build cache while staying in the root
echo "--- Cleaning old build... ---"
jupyter-book clean . --all

# 3. Build the site
echo "--- Building Jupyter Book... ---"
jupyter-book build .

# 4. Serve the site safely
if [ $? -eq 0 ]; then
    echo "--- Serving site at http://localhost:8000 ---"
    echo "--- (Server is running from _build/html, but script stays in root) ---"

    # Using ( ) creates a subshell. When the server stops,
    # you are automatically back in the project root.
    (cd _build/html && python3 -m http.server 8000)

    echo "--- Server stopped. You are back in $(pwd) ---"
else
    echo "Build failed."
    exit 1
fi
