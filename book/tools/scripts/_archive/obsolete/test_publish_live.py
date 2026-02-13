#!/usr/bin/env python3
"""
Test script for the new publish-live workflow.

This script helps verify that the PDF handling is working correctly
without committing the PDF to git.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_git_status():
    """Check if there are any uncommitted changes"""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'],
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Error checking git status: {e}")
        return None

def check_pdf_in_git():
    """Check if the PDF is being tracked by git"""
    try:
        result = subprocess.run(['git', 'ls-files', 'assets/Machine-Learning-Systems.pdf'],
                              capture_output=True, text=True)
        return result.stdout.strip() != ""
    except subprocess.CalledProcessError:
        return False

def check_pdf_exists():
    """Check if the PDF exists in assets"""
    pdf_path = Path("assets/Machine-Learning-Systems.pdf")
    return pdf_path.exists()

def check_gitignore():
    """Check if the PDF is properly ignored in .gitignore"""
    try:
        with open(".gitignore", "r") as f:
            content = f.read()
            return "assets/Machine-Learning-Systems.pdf" in content
    except FileNotFoundError:
        return False

def main():
    """Main test function"""
    print("🧪 Testing PDF handling in publish-live workflow...")
    print()

    # Check 1: Git status
    print("1️⃣ Checking git status...")
    git_status = check_git_status()
    if git_status:
        print(f"   📊 Git status: {git_status}")
        if "assets/Machine-Learning-Systems.pdf" in git_status:
            print("   ❌ PDF is showing in git status!")
        else:
            print("   ✅ PDF not in git status")
    else:
        print("   ⚠️ Could not check git status")

    # Check 2: PDF in git tracking
    print("\n2️⃣ Checking if PDF is tracked by git...")
    if check_pdf_in_git():
        print("   ❌ PDF is being tracked by git!")
    else:
        print("   ✅ PDF is not tracked by git")

    # Check 3: PDF exists
    print("\n3️⃣ Checking if PDF exists in assets...")
    if check_pdf_exists():
        print("   ✅ PDF exists in assets/")
        size = Path("assets/Machine-Learning-Systems.pdf").stat().st_size
        print(f"   📊 PDF size: {size / (1024*1024):.1f} MB")
    else:
        print("   ⚠️ PDF not found in assets/")

    # Check 4: Gitignore configuration
    print("\n4️⃣ Checking .gitignore configuration...")
    if check_gitignore():
        print("   ✅ PDF is properly ignored in .gitignore")
    else:
        print("   ❌ PDF not found in .gitignore")

    print("\n📋 Summary:")
    print("   - PDF should exist in assets/ for download")
    print("   - PDF should NOT be tracked by git")
    print("   - PDF should be in .gitignore")
    print("   - PDF will be uploaded to GitHub Release assets")
    print("\n🔗 PDF will be available at:")
    print("   - Direct: https://mlsysbook.ai/assets/Machine-Learning-Systems.pdf")
    print("   - Release: https://github.com/harvard-edge/cs249r_book/releases")

if __name__ == "__main__":
    main()
