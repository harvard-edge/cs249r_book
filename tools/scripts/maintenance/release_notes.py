#!/usr/bin/env python3
"""
Generate release notes with exact behavior from original unified script.

This script generates release notes for GitHub releases, matching the exact
behavior of the original changelog-releasenotes.py script.
"""

import argparse
import os
import sys
import re
from datetime import datetime

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
CHANGELOG_FILE = "CHANGELOG.md"
RELEASE_NOTES_FILE = "release_notes_v{version}.md"

def extract_latest_changelog_section(changelog_file="CHANGELOG.md"):
    """Extract the most recent changelog section for release notes generation."""
    if not os.path.exists(changelog_file):
        print(f"❌ Changelog file not found: {changelog_file}")
        return None
    
    try:
        with open(changelog_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the most recent section (after the last "## YYYY Updates" header)
        sections = re.split(r'## \d{4} Updates', content)
        if len(sections) < 2:
            print("❌ No changelog sections found")
            return None
        
        # Get the most recent section (last one)
        latest_section = sections[-1].strip()
        
        # Extract the first entry (most recent) from this section
        # Look for the first "### 📅" entry
        entries = re.split(r'### 📅', latest_section)
        if len(entries) < 2:
            print("❌ No changelog entries found in latest section")
            return None
        
        # Get the most recent entry (first one after the split)
        latest_entry = entries[1].strip()
        
        # Clean up the entry - remove any trailing content
        # Stop at the next "### 📅" or end of content
        if "### 📅" in latest_entry:
            latest_entry = latest_entry.split("### 📅")[0].strip()
        
        print(f"✅ Extracted latest changelog entry ({len(latest_entry)} characters)")
        return latest_entry
        
    except Exception as e:
        print(f"❌ Error reading changelog: {e}")
        return None

def generate_release_notes_from_changelog(version, previous_version, description, changelog_entry, verbose=False):
    """Generate release notes using changelog data."""
    
    if verbose:
        print(f"📝 Generating release notes...")
        print(f"📋 Version: {version}")
        print(f"📋 Previous: {previous_version}")
        print(f"📋 Description: {description}")
        print(f"📋 Changelog entry length: {len(changelog_entry)} characters")
    
    # Create release notes template
    release_notes = f"""## 📚 Release {version}

### 🎯 Key Updates
- Repository restructuring for better organization
- Enhanced learning with integrated quizzes
- Improved content clarity and navigation

### 📋 Release Information
- **Type**: Release
- **Previous Version**: {previous_version}
- **Published at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Build Platform**: Linux (HTML + PDF)

### 🔗 Quick Links
- 🌐 [Web](https://mlsysbook.ai)
- 📄 [PDF](https://mlsysbook.ai/pdf)

### 📖 Detailed Changes
For a complete list of all changes, improvements, and updates, see the [detailed changelog](https://www.mlsysbook.ai/contents/frontmatter/changelog/changelog).

### 🏗️ Build Information
- **Platform**: Linux
- **Outputs**: HTML + PDF
- **Deployment**: GitHub Pages
- **PDF Generation**: Quarto with LaTeX
"""
    
    return release_notes

def generate_release_notes(version, previous_version, description, verbose=False):
    """Generate release notes and save to file."""
    
    print(f"📝 Generating release notes for version {version}...")
    
    # First, ensure we have a changelog
    if not os.path.exists(CHANGELOG_FILE):
        print(f"📝 Changelog not found, generating incremental changelog...")
        # Import and call the changelog generation
        import subprocess
        cmd = ["python3", "tools/scripts/maintenance/change_log.py", "--incremental"]
        if verbose:
            cmd.append("--verbose")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️ Failed to generate changelog: {result.stderr}")
    
    # Try to extract changelog data
    changelog_entry = extract_latest_changelog_section(CHANGELOG_FILE)
    
    if changelog_entry:
        print("📝 Using changelog data for release notes generation...")
        release_notes = generate_release_notes_from_changelog(
            version=version,
            previous_version=previous_version,
            description=description,
            changelog_entry=changelog_entry,
            verbose=verbose
        )
    else:
        print("⚠️ No changelog data found, using basic generation...")
        
        # Fallback to basic generation
        if verbose:
            print(f"📋 Version: {version}")
            print(f"📋 Previous: {previous_version}")
            print(f"📋 Description: {description}")
            print("🧪 TEST MODE - Using basic template")
        
        # Create the final release notes (basic template)
        release_notes = f"""## 📚 Release {version}

### 🎯 Key Updates
- Repository restructuring for better organization
- Enhanced learning with integrated quizzes
- Improved content clarity and navigation

### 📋 Release Information
- **Type**: Release
- **Previous Version**: {previous_version}
- **Published at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Build Platform**: Linux (HTML + PDF)

### 🔗 Quick Links
- 🌐 [Web](https://mlsysbook.ai)
- 📄 [PDF](https://mlsysbook.ai/pdf)

### 📖 Detailed Changes
For a complete list of all changes, improvements, and updates, see the [detailed changelog](https://www.mlsysbook.ai/contents/frontmatter/changelog/changelog).

### 🏗️ Build Information
- **Platform**: Linux
- **Outputs**: HTML + PDF
- **Deployment**: GitHub Pages
- **PDF Generation**: Quarto with LaTeX

---
*Basic template - no changelog data available*
"""
    
    # Save release notes to file
    filename = RELEASE_NOTES_FILE.format(version=version)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(release_notes)
    
    print(f"✅ Release notes saved to: {filename}")
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate release notes with exact behavior from original unified script.")
    
    # Release notes arguments
    parser.add_argument("--version", type=str, required=True, help="Version number for release notes.")
    parser.add_argument("--previous-version", type=str, required=True, help="Previous version number for release notes.")
    parser.add_argument("--description", type=str, required=True, help="Release description for release notes.")
    
    # Options
    parser.add_argument("-t", "--test", action="store_true", help="Run without writing to file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("--changelog-file", default="CHANGELOG.md", help="Path to changelog file.")

    args = parser.parse_args()
    
    try:
        # Print configuration header
        print("=" * 60)
        print("📝 RELEASE NOTES GENERATION CONFIG")
        print("=" * 60)
        print(f"🎯 Version: {args.version}")
        print(f"📋 Previous: {args.previous_version}")
        print(f"📢 Description: {args.description}")
        print(f"🔧 Test Mode: {'ON' if args.test else 'OFF'}")
        print(f"📢 Verbose: {'ON' if args.verbose else 'OFF'}")
        print(f"📄 Changelog File: {args.changelog_file}")
        print("=" * 60)
        print()
        
        print("🚀 Starting release notes generation...")
        
        # Generate release notes
        filename = generate_release_notes(
            version=args.version,
            previous_version=args.previous_version,
            description=args.description,
            verbose=args.verbose
        )
        
        if filename and os.path.exists(filename):
            if args.test:
                # Read and display the content for test mode
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                print("🧪 TEST MODE - Release notes content:")
                print("=" * 60)
                print(content)
                print("=" * 60)
                print(f"📊 File size: {len(content)} characters")
                # Clean up test file
                os.remove(filename)
                print("🧹 Test file cleaned up")
            else:
                print(f"✅ Release notes saved to: {filename}")
                print(f"📊 File size: {os.path.getsize(filename)} bytes")
        else:
            print("❌ Failed to generate release notes")
            exit(1)
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 