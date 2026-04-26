#!/bin/sh
# TeX Live install-tl download + base install (one layer, POSIX sh for dash).
# Keep logic out of the Dockerfile RUN string: buildx/sh can mangle $ vs $$( command substitution.
rev="${1:-0}"
printf '%s\n' "🔄 TeX Live layer rev $rev: download + base install"
printf '%s\n' "🔄 Downloading TeX Live installer, ~4MB, with mirror fallbacks..."

SUCCESS=false
for mirror in \
    "https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz" \
    "https://mirrors.mit.edu/CTAN/systems/texlive/tlnet/install-tl-unx.tar.gz" \
    "https://ctan.math.washington.edu/tex-archive/systems/texlive/tlnet/install-tl-unx.tar.gz" \
    "https://mirrors.rit.edu/CTAN/systems/texlive/tlnet/install-tl-unx.tar.gz" \
    "https://mirror.las.iastate.edu/tex-archive/systems/texlive/tlnet/install-tl-unx.tar.gz"
do
    echo "🔄 Trying mirror: $mirror"
    if wget --timeout=30 --tries=2 -O /tmp/install-tl-unx.tar.gz "$mirror"; then
        echo "✅ Download successful from: $mirror"
        SUCCESS=true
        break
    else
        echo "❌ Failed to download from: $mirror"
        rm -f /tmp/install-tl-unx.tar.gz
    fi
done

if [ "$SUCCESS" = false ]; then
    echo "❌ All mirrors failed, cannot proceed"
    exit 1
fi
echo "📥 Download completed"
echo "📊 Downloaded file size:"
ls -lh /tmp/install-tl-unx.tar.gz

echo "📦 Extracting TeX Live installer..."
cd /tmp
tar -xzf install-tl-unx.tar.gz
echo "📦 Extraction completed"
echo "📊 Extracted files:"
ls -la /tmp/install-tl-*
echo "✅ TeX Live installer ready"
set -- /tmp/install-tl-*/
if [ ! -d "$1" ]; then
    echo "❌ No /tmp/install-tl-* after extract"
    exit 1
fi
TLVER=$(basename "$1")
# install-tl-YYYYMMDD → year at bytes 12–15
TLYY=$(printf '%s\n' "$TLVER" | cut -c12-15)
case "$TLYY" in
    2025) TLNET="https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2025/tlnet-final" ;;
    2026) TLNET="https://mirror.ctan.org/systems/texlive/tlnet" ;;
    2027) TLNET="https://mirror.ctan.org/systems/texlive/tlnet" ;;
    *) echo "❌ Unsupported TeX install-tl id"; echo "  TLVER=$TLVER"; echo "  TLYY=$TLYY"; exit 1 ;;
esac
echo "📍 TeX tlpdb: $TLVER at $TLNET"
mkdir -p /opt
printf '%s\n' "$TLNET" > /opt/mlsysbook-texlive-tlnet-url

echo "🔧 Creating TeX Live installation profile..."
echo "selected_scheme scheme-medium" > /tmp/texlive.profile
echo "tlpdbopt_install_docfiles 0" >> /tmp/texlive.profile
echo "tlpdbopt_install_srcfiles 0" >> /tmp/texlive.profile
echo "TEXDIR /usr/local/texlive" >> /tmp/texlive.profile
echo "TEXMFCONFIG /usr/local/texlive/texmf-config" >> /tmp/texlive.profile
echo "TEXMFHOME /usr/local/texlive/texmf-home" >> /tmp/texlive.profile
echo "TEXMFLOCAL /usr/local/texlive/texmf-local" >> /tmp/texlive.profile
echo "TEXMFSYSCONFIG /usr/local/texlive/texmf-config" >> /tmp/texlive.profile
echo "TEXMFSYSVAR /usr/local/texlive/texmf-var" >> /tmp/texlive.profile
echo "TEXMFVAR /usr/local/texlive/texmf-var" >> /tmp/texlive.profile
echo "📄 Profile created"
echo "📊 Profile contents:"
cat /tmp/texlive.profile

echo "🔄 Installing TeX Live base system with retry logic..."
INSTALL_SUCCESS=false
_inst="$1/install-tl"
if [ ! -f "$_inst" ]; then
    echo "❌ Missing $_inst"
    exit 1
fi
for attempt in 1 2 3; do
    echo "🔄 Installation attempt $attempt/3..."
    if "$_inst" -repository "$TLNET" --profile=/tmp/texlive.profile; then
        echo "✅ TeX Live installation successful on attempt $attempt"
        INSTALL_SUCCESS=true
        break
    else
        echo "❌ Installation attempt $attempt failed"
        if [ "$attempt" -lt 3 ]; then
            echo "⏳ Waiting 10 seconds before retry..."
            sleep 10
        fi
    fi
done

if [ "$INSTALL_SUCCESS" = false ]; then
    echo "❌ All installation attempts failed"
    echo "📋 Checking for partial installation..."
    ls -la /usr/local/texlive/ || echo "No TeX Live directory found"
    exit 1
fi
echo "📦 TeX Live base system installed"
