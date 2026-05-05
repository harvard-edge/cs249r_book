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

# install-tl ran minutes ago against a tlnet snapshot; tlnet rolls forward
# continuously, so by the time we reach 'tlmgr install collection-X' the local
# tlmgr is often older than the remote tlpdb and tlmgr refuses with
# "Local TL version is incompatible with the repository". Bring tlmgr in sync
# first. Failure here is non-fatal — if the remote already matches, the update
# is a no-op; if the network is flaky, the per-collection retry loop still gets
# a chance.
echo "🔄 Syncing tlmgr to remote tlpdb (tlmgr update --self)..."
if tlmgr update --self; then
  echo "✅ tlmgr is in sync with remote tlpdb"
else
  echo "⚠️ tlmgr --self update failed; collection install may still recover"
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
    printf '%s\n' "📦 [$i/$collection_count] Processing $collection..."
    if command -v tlmgr >/dev/null 2>&1; then
      # install-tl already installs most of these via scheme-medium (basic,
      # fontsrecommended, fontutils, latex, latexrecommended, luatex,
      # pictures, ...). Calling 'tlmgr install <already-present>' against
      # mirror.ctan.org's random-mirror redirect can hit a stale mirror that
      # refuses with a silent version-mismatch exit 1 — the failure mode that
      # took down the 2026-05-04 16:23 build for collection-fontsrecommended.
      # Query the local tlpdb (no network) first and skip if already installed;
      # only collections that genuinely need network install (e.g. fontsextra,
      # latexextra) reach the tlmgr install retry loop.
      # Match 'installed: Yes' specifically. tlmgr prints 'package: NAME' for
      # *both* installed and not-installed lookups (verified against TL 2026
      # locally: a not-installed lookup still prints 'package: NAME' followed
      # by 'installed: No'), so grepping '^package:' would falsely skip
      # everything — including the collections that genuinely need network
      # install. Anchor on 'installed: Yes' instead.
      if tlmgr info --only-installed "$collection" 2>/dev/null | grep -q '^installed:.*Yes'; then
        printf '%s\n' "✅ $collection already installed locally (no network call)"
        i=$(expr "$i" + 1)
        continue
      fi
      success=false
      for retry in 1 2; do
        # Capture tlmgr stdout+stderr so its actual error message reaches the
        # CI log. The previous 'if tlmgr install ...; then' form left only
        # "Attempt N failed" with no diagnostic, which masked a silent
        # collection-fontsextra failure that took 3 hours to surface as a
        # Vol II PDF build error.
        tlmgr_log=$(mktemp)
        tlmgr install "$collection" >"$tlmgr_log" 2>&1
        tlmgr_status=$?
        cat "$tlmgr_log"
        rm -f "$tlmgr_log"
        if [ "$tlmgr_status" -eq 0 ]; then
          printf '%s\n' "✅ Successfully installed $collection"
          success=true
          break
        else
          printf '%s\n' "❌ Attempt $retry failed for $collection (exit $tlmgr_status)"
          if [ "$retry" -lt 2 ]; then
            echo "⏳ Retrying in 5 seconds..."
            sleep 5
          fi
        fi
      done
      if [ "$success" = "false" ]; then
        echo "❌ Failed to install $collection after retries"
        failed_packages="$failed_packages $collection"
      fi
    else
      echo "❌ tlmgr not available, cannot install $collection"
      failed_packages="$failed_packages $collection"
    fi
    i=$(expr "$i" + 1)
    ;;
  esac
done < /tmp/tl_packages

if [ -n "$failed_packages" ]; then
  echo "❌ TeX Live collections failed to install:$failed_packages"
  echo "📋 Every collection in tl_packages is required by the book PDF build."
  echo "📋 Failing the container build now rather than publishing a broken"
  echo "📋 :latest image (collection-fontsextra silently dropping newpx caused"
  echo "📋 a 3-hour-delayed Vol II PDF build failure on 2026-05-04)."
  exit 1
fi
echo "✅ TeX Live packages installation completed"
