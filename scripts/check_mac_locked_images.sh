#!/bin/bash

# Exit silently if not running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
  exit 0
fi

found=0

# Loop over staged image files only
while IFS= read -r file; do
  if xattr "$file" 2>/dev/null | grep -q "com.apple.quarantine"; then
    echo "❌ Quarantined image detected: $file"
    found=1
  fi
done < <(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(png|jpg|jpeg|pdf)$')

if [ "$found" -eq 1 ]; then
  echo ""
  echo "🛑 Please remove quarantine flags using:"
  echo "    xattr -d com.apple.quarantine <file>"
  exit 1
fi

exit 0
