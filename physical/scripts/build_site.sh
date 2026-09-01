#!/usr/bin/env bash
# Assemble the full Physical AI site into ./_site
#
#   _site/         <- online interactive textbook (HTML)
#   _site/pdf/     <- dedicated / hidden PDF endpoint (Physical-AI.pdf)
#
set -euo pipefail
cd "$(dirname "$0")/.."

SITE_DOMAIN="${PAI_SITE_DOMAIN:-physical.mlsysbook.ai}"
BUILD_DIR="book/_build"
SITE_DIR="_site"

echo "==> Cleaning old site build"
rm -rf "$SITE_DIR"
mkdir -p "$SITE_DIR/pdf"

echo "==> Rendering HTML textbook"
quarto render book --to html
cp -r "$BUILD_DIR"/* "$SITE_DIR"/

if [[ "${SKIP_PDF:-0}" != "1" ]]; then
  echo "==> Rendering PDF textbook"
  quarto render book --to pdf
fi

if [[ -f "$BUILD_DIR/Physical-AI.pdf" ]]; then
  echo "==> Copying PDF to hidden /pdf/ route"
  cp "$BUILD_DIR/Physical-AI.pdf" "$SITE_DIR/pdf/Physical-AI.pdf"
  
  # Create a clean direct viewer / redirect page at /pdf/index.html
  cat << 'HTMLEOF' > "$SITE_DIR/pdf/index.html"
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="0; url=Physical-AI.pdf">
  <title>Physical AI — PDF Edition</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #0f172a; color: #f8fafc; }
    .card { text-align: center; background: #1e293b; padding: 2.5rem; border-radius: 8px; box-shadow: 0 10px 25px rgba(0,0,0,0.3); }
    a.btn { display: inline-block; margin-top: 1.5rem; background: #A51C30; color: white; padding: 0.75rem 1.5rem; border-radius: 4px; text-decoration: none; font-weight: 600; }
    a.btn:hover { background: #821424; }
  </style>
</head>
<body>
  <div class="card">
    <h2>Physical AI: Systems Architecture</h2>
    <p>If your download does not start automatically, click below:</p>
    <a class="btn" href="Physical-AI.pdf" download>Download Physical-AI.pdf</a>
  </div>
</body>
</html>
HTMLEOF
fi

echo "==> Creating CNAME, .nojekyll, and robots.txt (noindex)"
echo "$SITE_DOMAIN" > "$SITE_DIR/CNAME"
touch "$SITE_DIR/.nojekyll"

cat << 'ROBOTSEOF' > "$SITE_DIR/robots.txt"
User-agent: *
Disallow: /
ROBOTSEOF

echo "==> Site build complete in $SITE_DIR"
