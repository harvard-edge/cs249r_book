#!/bin/bash
# =============================================================================
# Quick Link Check for The Blueprint (Instructor Site)
# =============================================================================
# Usage: cd instructors && ./check-links.sh
#
# Checks all .qmd files for:
#   1. Internal file references that don't exist
#   2. External URLs that return non-200 status codes
#
# Requires: python3
# Optional: lychee (brew install lychee) for full external URL checking
# =============================================================================

set -euo pipefail

echo "🔗 Checking links in instructor site..."
echo ""

# ── 1. Internal file references ──────────────────────────────────────────────
echo "── Internal References ──"

python3 - <<'PYEOF'
import re, os, glob, sys

# Check markdown links: [text](target)
link_pattern = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')
# Check image refs: ![alt](path)
img_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

fails = []
warnings = []

for qmd in sorted(glob.glob("*.qmd")):
    with open(qmd) as f:
        for lineno, line in enumerate(f, 1):
            for pattern in [link_pattern, img_pattern]:
                for m in pattern.finditer(line):
                    target = m.group(2).split("{")[0].split("#")[0].strip()

                    # Skip external URLs, anchors, mailto
                    if target.startswith(("http://", "https://", "#", "mailto:")):
                        continue
                    if not target:
                        continue

                    # Check if file exists
                    if not os.path.isfile(target):
                        fails.append(f"  {qmd}:{lineno} → {target}")

if fails:
    print(f"❌ {len(fails)} broken internal reference(s):")
    for f in fails:
        print(f)
else:
    print("✅ All internal references valid")

print("")
PYEOF

# ── 2. External URLs (quick check) ──────────────────────────────────────────
echo "── External URLs ──"

if command -v lychee &> /dev/null; then
    lychee --no-progress --exclude-path node_modules --max-concurrency 5 \
           --accept 200,403 \
           --exclude 'localhost' --exclude '127.0.0.1' \
           ./*.qmd || echo "⚠️  Some external links may be broken"
else
    echo "ℹ️  Install lychee for full external URL checking: brew install lychee"
    echo "   Running basic URL extraction instead..."
    echo ""

    python3 - <<'PYEOF'
import re, glob

url_pattern = re.compile(r'https?://[^\s\)>"]+')
urls = set()

for qmd in sorted(glob.glob("*.qmd")):
    with open(qmd) as f:
        for line in f:
            for m in url_pattern.finditer(line):
                urls.add(m.group(0).rstrip('.,;:'))

print(f"Found {len(urls)} unique external URLs across all pages.")
print("Install lychee to validate them: brew install lychee")
PYEOF
fi

echo ""
echo "Done."
