#!/usr/bin/env python3
"""
Modify announcement banner for development preview deployment.

This script modifies the Quarto announcement banner in HTML files to:
1. Make it non-dismissible
2. Add a prominent "DEVELOPMENT PREVIEW" banner at the top
3. Style it with appropriate warning colors

Used by the deploy-preview GitHub Actions workflow.
"""

import os
import sys
import re
from pathlib import Path
from typing import List
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo


def find_html_files_with_announcement(directory: Path) -> List[Path]:
    """Find all HTML files that contain quarto-announcement."""
    html_files = []

    for html_file in directory.rglob("*.html"):
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'quarto-announcement' in content:
                    html_files.append(html_file)
        except Exception as e:
            print(f"⚠️  Warning: Could not read {html_file}: {e}")

    return html_files


def modify_announcement_banner(file_path: Path, commit_hash: str = None, commit_short: str = None) -> bool:
    """
    Modify the announcement banner in an HTML file for development preview.

    Returns True if modifications were made, False otherwise.
    """
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 1. Make announcement non-dismissible by hiding the close button
        content = re.sub(
            r'<i class="bi bi-x-lg quarto-announcement-action"[^>]*>.*?</i>',
            '<i class="bi bi-x-lg quarto-announcement-action" style="display: none;"></i>',
            content,
            flags=re.DOTALL
        )

        # 2. Remove dismissible functionality
        content = re.sub(r'data-bs-dismiss="alert"', '', content)

        # 3. Add development preview text at the beginning of the existing content
        # Get current Eastern Time (EST/EDT)
        try:
            # Use timezone-aware datetime with Eastern Time
            eastern = ZoneInfo("America/New_York")
            eastern_now = datetime.now(eastern)
            timestamp = eastern_now.strftime("%Y-%m-%d %H:%M %Z")
        except Exception:
            # Fallback to UTC if timezone fails
            utc_now = datetime.utcnow()
            timestamp = utc_now.strftime("%Y-%m-%d %H:%M UTC")

        commit_info = ""
        if commit_hash and commit_short:
            commit_info = f''' Built from dev@<code style="background: rgba(0,0,0,0.1); padding: 2px 4px; border-radius: 3px; font-size: 0.9em;">{commit_short}</code> • {timestamp}'''
        elif commit_short:
            commit_info = f''' Built from commit <code style="background: rgba(0,0,0,0.1); padding: 2px 4px; border-radius: 3px; font-size: 0.9em;">{commit_short}</code> • {timestamp}'''
        else:
            commit_info = f''' • {timestamp}'''

        dev_text = f'''<p style="margin: 0 0 12px 0; padding: 8px 12px; background: rgba(255,193,7,0.2); border: 1px solid #ffc107; border-radius: 4px; font-weight: 600;"><i class="bi bi-exclamation-triangle-fill" style="margin-right: 6px; color: #856404;"></i><strong>🚧 DEVELOPMENT PREVIEW</strong> - Built from dev@<code style="background: rgba(0,0,0,0.1); padding: 2px 4px; border-radius: 3px; font-size: 0.9em;">{commit_short or "unknown"}</code> • {timestamp} • <a href="https://mlsysbook.ai" style="color: #856404; text-decoration: underline;"><em>Stable version →</em></a></p>'''

        # Insert the dev text right after the opening of quarto-announcement-content
        content = re.sub(
            r'(<div class="quarto-announcement-content">\s*)',
            f'\\1{dev_text}\n',
            content
        )

        # 4. Keep the original alert-primary styling - don't change it
        # The dev preview box has its own styling, so we leave the main announcement unchanged

        # Check if any changes were made
        if content != original_content:
            # Write the modified content back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"❌ Error modifying {file_path}: {e}")
        return False


def main():
    """Main function to process HTML files."""
    parser = argparse.ArgumentParser(description='Modify announcement banners for development preview')
    parser.add_argument('directory', help='Directory containing HTML files to process')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--commit-hash', help='Full commit hash for the build')
    parser.add_argument('--commit-short', help='Short commit hash for display')

    args = parser.parse_args()

    directory = Path(args.directory)

    if not directory.exists():
        print(f"❌ Error: Directory {directory} does not exist")
        sys.exit(1)

    print("🔧 Modifying announcement banners for development preview...")

    # Find HTML files with announcements
    html_files = find_html_files_with_announcement(directory)

    if not html_files:
        print("ℹ️  No HTML files with announcements found")
        return

    print(f"📝 Found {len(html_files)} HTML files with announcements")

    modified_count = 0

    for html_file in html_files:
        if args.verbose:
            print(f"  📄 Processing: {html_file.relative_to(directory)}")

        if modify_announcement_banner(html_file, args.commit_hash, args.commit_short):
            modified_count += 1
            if args.verbose:
                print(f"  ✅ Modified: {html_file.relative_to(directory)}")
        elif args.verbose:
            print(f"  ⏭️  No changes needed: {html_file.relative_to(directory)}")

    print(f"✅ Successfully modified {modified_count} files")

    if modified_count > 0:
        print("🎨 Development preview banner added to announcement sections")


if __name__ == "__main__":
    main()
