#!/usr/bin/env python3
"""
Setup script to install binder CLI in virtual environment
This allows using 'binder' command without './' when venv is active
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Install binder CLI in development mode"""
    project_root = Path(__file__).parent.parent  # Go up from book/ to repo root

    print("🔧 Setting up binder CLI in virtual environment...")

    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not in a virtual environment")
        print("   Consider activating your venv first: source .venv/bin/activate")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("❌ Setup cancelled")
            return 1

    try:
        # Install in development mode
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], cwd=project_root, check=True)

        print("✅ Binder CLI installed successfully!")
        print()
        print("📋 You can now use:")
        print("   binder help              # Global command (when venv active)")
        print("   ./binder help            # Local script (always works)")
        print()
        print("🎯 Both commands do the same thing - use whichever you prefer!")

        return 0

    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
