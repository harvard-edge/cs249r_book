#!/bin/bash
# TinyTorch activation helper
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    # On Apple Silicon, ensure arm64
    export TINYTORCH_ARCH="arm64"
    alias python='arch -arm64 .venv/bin/python3'
    alias pip='arch -arm64 .venv/bin/pip'
    source .venv/bin/activate
    echo "🔥 TinyTorch environment activated (arm64)"
else
    source .venv/bin/activate
    echo "🔥 TinyTorch environment activated"
fi

# Check if tito command is available, if not install package
if ! command -v tito &> /dev/null; then
    echo "📦 Installing TinyTorch CLI..."
    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        arch -arm64 .venv/bin/pip install -e . -q
    else
        pip install -e . -q
    fi
    echo "✅ TinyTorch CLI installed"
fi

echo "💡 Try: tito system health"
