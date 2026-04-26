#!/bin/sh
# Install TeX Live collection packages listed in /tmp/tl_packages (tlpdb collection-* lines).
# Extracted from Dockerfile so /bin/dash and BuildKit $ handling do not mangle a long RUN.
set +e

export PATH="/usr/local/texlive/bin/x86_64-linux:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

REPO_FILE="/opt/mlsysbook-texlive-tlnet-url"
if [ ! -f "$REPO_FILE" ]; then
  echo "❌ missing $REPO_FILE"
  exit 1
fi
TLMGR_REPO=$(tr -d '\n' < "$REPO_FILE")
if ! tlmgr option repository "$TLMGR_REPO"; then
  echo "❌ tlmgr option repository failed"
  exit 1
fi

printf '%s\n' "📊 Analyzing tl_packages file..."
collection_count=$(grep -c '^collection-' /tmp/tl_packages 2>/dev/null || true)
if [ -z "$collection_count" ] || [ "$collection_count" = "0" ]; then
  collection_count=0
fi
printf '%s\n' "📦 Found $collection_count TeX Live collections to install"
printf '%s\n' "🔄 Installing TeX Live collections with retry logic..."

i=1
failed_packages=""
while IFS= read -r collection; do
  case "$collection" in
  collection-*)
    printf '%s\n' "📦 [$i/$collection_count] Installing $collection..."
    if command -v tlmgr >/dev/null 2>&1; then
      success=false
      for retry in 1 2; do
        if tlmgr install "$collection"; then
          printf '%s\n' "✅ Successfully installed $collection"
          success=true
          break
        else
          printf '%s\n' "❌ Attempt $retry failed for $collection"
          if [ "$retry" -lt 2 ]; then
            echo "⏳ Retrying in 5 seconds..."
            sleep 5
          fi
        fi
      done
      if [ "$success" = "false" ]; then
        echo "⚠️ Failed to install $collection after retries, continuing..."
        failed_packages="$failed_packages $collection"
      fi
    else
      echo "⚠️ tlmgr not available, skipping $collection"
    fi
    i=$(expr "$i" + 1)
    ;;
  esac
done < /tmp/tl_packages

if [ -n "$failed_packages" ]; then
  echo "⚠️ Some packages failed to install:$failed_packages"
  echo "📋 This may not be critical for basic functionality"
fi
echo "✅ TeX Live packages installation completed"
