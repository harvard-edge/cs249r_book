#!/usr/bin/env python3
"""
MLSysBook Custom Extension Protection Checker

Verifies that our custom extensions are protected from accidental reinstallation
by checking that they use the mlsysbook/ namespace that won't conflict with
standard `quarto add` commands.
"""

import os
import sys
from pathlib import Path

def main():
    print("MLSysBook Custom Extension Protection Check")
    print("=" * 50)

    # Define expected custom extensions in mlsysbook-ext namespace
    protected_extensions = {
        "mlsysbook-ext/custom-numbered-blocks": {
            "type": "moved",
            "original": "ute/custom-numbered-blocks",
            "description": "Custom numbered blocks with MLSysBook styling"
        },
        "mlsysbook-ext/titlepage": {
            "type": "moved",
            "original": "nmfs-opensci/titlepage",
            "description": "Custom titlepage with MLSysBook branding"
        },
        "mlsysbook-ext/margin-video": {
            "type": "custom",
            "original": None,
            "description": "Custom margin video extension for YouTube embedding"
        }
    }

    # Base path for extensions
    extensions_dir = Path("book/_extensions")

    if not extensions_dir.exists():
        print(f"❌ Extensions directory not found: {extensions_dir}")
        sys.exit(1)

    print("🔒 Checking custom extension protection via mlsysbook-ext/ namespace...")

    # Check each protected extension
    all_protected = True
    found_extensions = []

    for ext_path, info in protected_extensions.items():
        full_path = extensions_dir / ext_path
        if full_path.exists():
            print(f"✅ Protected custom extension: {ext_path}")
            found_extensions.append(ext_path)
        else:
            print(f"❌ Missing protected extension: {ext_path}")
            all_protected = False

    # Check for old naming scheme (should be migrated)
    old_naming = [
        "ute-mlsysbook-custom/custom-numbered-blocks",
        "nmfs-opensci-mlsysbook-custom/titlepage",
        "margin-video-mlsysbook",
        "mlsysbook/custom-numbered-blocks",
        "mlsysbook/titlepage",
        "mlsysbook/margin-video"
    ]

    found_old = []
    for old_path in old_naming:
        if (extensions_dir / old_path).exists():
            found_old.append(old_path)
            print(f"⚠️  Warning: Found old naming scheme: {old_path}")
            print(f"   Consider migrating to mlsysbook-ext/ namespace")

    print()

    if all_protected and not found_old:
        print(f"✅ All {len(found_extensions)} custom extensions are properly protected!")
        print()
        print("🎉 Extension protection is intact!")
        if found_extensions:
            print()
            print("Protected extensions:")
            for ext in found_extensions:
                print(f"  - {ext}")
    else:
        print("❌ Extension protection issues detected!")
        if not all_protected:
            print("  - Some custom extensions are missing")
        if found_old:
                    print("  - Old naming scheme detected")
        print("  - Consider migrating to mlsysbook-ext/ namespace")
        sys.exit(1)

if __name__ == "__main__":
    main()
