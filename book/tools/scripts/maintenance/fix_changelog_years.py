#!/usr/bin/env python3
"""
Fix changelog year bucketing by ensuring proper year headers and organizing entries by year.
"""

import re
from datetime import datetime
from collections import defaultdict

def extract_date_from_entry(entry_text):
    """Extract date from a changelog entry."""
    # Look for date pattern like "### 📅 June 10 at 02:36 PM"
    date_match = re.search(r'### 📅 (.+?)(?:\n|$)', entry_text)
    if date_match:
        date_str = date_match.group(1)
        try:
            # Parse date like "June 10 at 02:36 PM"
            # Add current year since the date doesn't include year
            current_year = datetime.now().year

            # Special handling for months that indicate previous year
            month = date_str.split()[0]
            if month in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']:
                # If it's January and we're in 2025, it's likely 2024
                if month == 'January' and current_year == 2025:
                    year = 2024
                elif month in ['November', 'December'] and current_year == 2025:
                    year = 2024
                else:
                    year = current_year
            else:
                year = current_year

            date_with_year = f"{date_str} {year}"
            dt = datetime.strptime(date_with_year, "%B %d at %I:%M %p %Y")
            return dt
        except:
            pass
    return None

def clean_date_format(entry_text):
    """Clean up the date format in an entry."""
    # Replace "### 📅 Month Day at Time" with "### Abbreviated Month Day"
    month_abbrev = {
        'January': 'Jan', 'February': 'Feb', 'March': 'Mar', 'April': 'Apr',
        'May': 'May', 'June': 'Jun', 'July': 'Jul', 'August': 'Aug',
        'September': 'Sep', 'October': 'Oct', 'November': 'Nov', 'December': 'Dec'
    }

    def replace_month(match):
        full_date = match.group(1)
        for full_month, abbrev in month_abbrev.items():
            if full_date.startswith(full_month):
                return f"### {full_date.replace(full_month, abbrev)}"
        return f"### {full_date}"

    cleaned = re.sub(r'### 📅 (.+?) at \d+:\d+ [AP]M', replace_month, entry_text)
    return cleaned

def fix_changelog_years(changelog_file="CHANGELOG.md"):
    """Fix the changelog to have proper year bucketing."""

    print(f"🔧 Fixing year bucketing in {changelog_file}...")

    # Read the changelog
    with open(changelog_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the header and disclaimer
    lines = content.split('\n')
    header_lines = []
    content_lines = []

    # Separate header from content
    in_header = True
    for line in lines:
        if in_header and (line.startswith('# Changelog') or line.startswith(':::') or line.strip() == ''):
            header_lines.append(line)
        else:
            in_header = False
            content_lines.append(line)

    # Extract all date entries
    entries = []
    current_entry = []

    for line in content_lines:
        if line.startswith('### 📅'):
            if current_entry:
                entries.append('\n'.join(current_entry))
            current_entry = [line]
        elif line.startswith('## '):
            # Skip year headers for now
            continue
        else:
            if current_entry:
                current_entry.append(line)

    if current_entry:
        entries.append('\n'.join(current_entry))

    print(f"📊 Found {len(entries)} date entries")

    # Group entries by year
    entries_by_year = defaultdict(list)

    for entry in entries:
        dt = extract_date_from_entry(entry)
        if dt:
            year = dt.year
            # Clean up the date format
            cleaned_entry = clean_date_format(entry.strip())
            entries_by_year[year].append(cleaned_entry)
            print(f"  📅 {dt.strftime('%B %d, %Y')} → {year}")
        else:
            print(f"  ⚠️ Could not parse date for entry")

    print(f"📅 Years found: {sorted(entries_by_year.keys())}")

    # Build new content
    new_content = []

    # Add header
    new_content.extend(header_lines)

    # Add year sections in reverse chronological order
    for year in sorted(entries_by_year.keys(), reverse=True):
        year_header = f"## 📅 {year}"
        year_entries = "\n\n".join(entries_by_year[year])
        new_content.append(f"{year_header}\n\n{year_entries}")

    # Write the fixed changelog
    fixed_content = "\n".join(new_content)

    # Backup original
    backup_file = f"{changelog_file}.backup"
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"💾 Backup saved to {backup_file}")

    # Write fixed content
    with open(changelog_file, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print(f"✅ Fixed changelog written to {changelog_file}")
    print(f"📊 Years organized: {sorted(entries_by_year.keys(), reverse=True)}")

    return entries_by_year

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix changelog year bucketing")
    parser.add_argument("--file", default="CHANGELOG.md", help="Changelog file to fix")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")

    args = parser.parse_args()

    if args.dry_run:
        print("🧪 DRY RUN MODE - No changes will be made")
        # Read and analyze without writing
        with open(args.file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Just show what years would be found
        lines = content.split('\n')
        entries = []
        current_entry = []

        for line in lines:
            if line.startswith('### 📅'):
                if current_entry:
                    entries.append('\n'.join(current_entry))
                current_entry = [line]
            elif line.startswith('## '):
                continue
            else:
                if current_entry:
                    current_entry.append(line)

        if current_entry:
            entries.append('\n'.join(current_entry))

        years_found = set()
        for entry in entries:
            dt = extract_date_from_entry(entry)
            if dt:
                years_found.add(dt.year)

        print(f"📅 Would organize entries into years: {sorted(years_found, reverse=True)}")
    else:
        fix_changelog_years(args.file)
