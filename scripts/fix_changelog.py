#!/usr/bin/env python3
"""
Changelog Format Converter
Converts existing changelog from old format to new organized format without OpenAI calls.
Preserves exact original order within each entry.
"""

import re
import sys
from pathlib import Path

def categorize_entry(entry_text):
    """Categorize a changelog entry based on simple title patterns."""
    # Extract just the title part (everything between ** markers)
    if '**' in entry_text:
        title_match = re.search(r'\*\*(.*?)\*\*', entry_text)
        if title_match:
            title = title_match.group(1).strip()
        else:
            title = entry_text.strip()
    else:
        title = entry_text.strip()
    
    # Simple rules based on your structure:
    
    # 1. If it starts with "Lab:" -> lab
    if title.startswith('Lab:'):
        return 'lab'
    
    # 2. If it starts with "Chapter X:" -> chapter  
    if re.match(r'^Chapter \d+:', title):
        return 'chapter'
    
    # 3. If it contains "PhD" (the survival guide) -> appendix
    if 'phd' in title.lower() or 'survival' in title.lower():
        return 'appendix'
    
    # 4. Everything else -> frontmatter (About, Foreword, Acknowledgements, Socratiq, etc.)
    return 'frontmatter'

def parse_single_entry(entry_text):
    """Parse a single changelog entry and extract date + items in exact order."""
    lines = entry_text.strip().split('\n')
    
    # Find the date line
    date_line = None
    for line in lines:
        if 'Published on' in line and '📅' in line:
            date_line = line.strip()
            if not date_line.startswith('### '):
                date_line = f"### {date_line}"
            break
    
    if not date_line:
        return None, []
    
    # Extract all bullet points in exact order, skipping HTML tags
    items = []
    current_item = ""
    inside_details = False
    
    for line in lines:
        line = line.strip()
        
        # Skip HTML structure lines
        if any(tag in line for tag in ['<details', '</details>', '<summary>', '</summary>']):
            inside_details = '<details' in line or '<summary>' in line
            continue
            
        # Skip lines that are just summary headers with emoji icons
        if line.startswith('**') and any(emoji in line for emoji in ['📄', '📖', '🧑‍💻', '📚']) and line.endswith('**'):
            continue
            
        if line.startswith('- ') or line.startswith('* '):
            if current_item:
                items.append(current_item.strip())
            current_item = line[2:]  # Remove bullet point
        elif line and not line.startswith('#') and '📅' not in line and current_item and not inside_details:
            # Continuation of previous item
            current_item += " " + line
    
    # Add the last item
    if current_item:
        items.append(current_item.strip())
    
    return date_line, items

def organize_items_preserving_order(items):
    """Group items by category while preserving exact original order within each category."""
    # Create ordered lists for each category
    frontmatter_items = []
    chapter_items = []
    lab_items = []
    appendix_items = []
    
    print(f"      🔍 Categorizing {len(items)} items:")
    
    # Go through items in original order and place them in appropriate category lists
    for i, item in enumerate(items):
        category = categorize_entry(item)
        
        # Extract title for debug
        title_match = re.search(r'\*\*(.*?)\*\*', item)
        title = title_match.group(1) if title_match else item[:50] + "..."
        
        print(f"        {i+1:2d}. '{title}' -> {category}")
        
        if category == 'frontmatter':
            frontmatter_items.append(item)
        elif category == 'lab':
            lab_items.append(item)
        elif category == 'appendix':
            appendix_items.append(item)
        else:  # chapter
            chapter_items.append(item)
    
    return {
        'frontmatter': frontmatter_items,
        'chapter': chapter_items,
        'lab': lab_items,
        'appendix': appendix_items
    }

def format_new_entry(date_line, categories, is_latest=False):
    """Format a new changelog entry with organized sections."""
    details_state = "open" if is_latest else ""
    
    entry = f"{date_line}\n\n"
    
    # Add sections in order: Frontmatter, Chapters, Labs, Appendix
    section_configs = [
        ('frontmatter', '📄 Frontmatter'),
        ('chapter', '📖 Chapter Updates'),
        ('lab', '🧑‍💻 Lab Updates'),
        ('appendix', '📚 Appendix')
    ]
    
    sections_added = 0
    for category_key, section_title in section_configs:
        if categories[category_key]:
            entry += f"<details {details_state}>\n"
            entry += f"<summary>**{section_title}**</summary>\n\n"
            
            for item in categories[category_key]:
                # Ensure item starts with - 
                if not item.startswith('-'):
                    item = f"- {item}"
                elif not item.startswith('- '):
                    item = f"- {item[1:].strip()}"
                entry += f"{item}\n"
            
            entry += "\n</details>\n\n"
            sections_added += 1
    
    return entry.rstrip() + "\n"

def convert_changelog(input_file, output_file=None):
    """Convert existing changelog to new format."""
    if not Path(input_file).exists():
        print(f"Error: File {input_file} not found!")
        return False
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📖 Reading file: {input_file}")
    
    new_content = ""
    
    # Split by year headers first
    year_sections = re.split(r'^## (\d{4}) Changes\s*$', content, flags=re.MULTILINE)
    
    if len(year_sections) < 2:
        print("⚠️  No year sections found. Trying to process as single section...")
        # Try to process entire content as one section
        year_sections = ['', '2025', content]  # Default to 2025
    
    # Process each year section
    for i in range(1, len(year_sections), 2):
        year = year_sections[i]
        year_content = year_sections[i + 1] if i + 1 < len(year_sections) else ""
        
        print(f"\n📅 Processing {year} changes...")
        new_content += f"## {year} Changes\n\n"
        
        # Split individual entries by date headers
        # Look for lines that start with ### 📅 Published on or just 📅 Published on
        entry_pattern = r'(?=### 📅 Published on|(?:^|\n)📅 Published on)'
        entry_chunks = re.split(entry_pattern, year_content)
        
        # Filter out empty chunks
        entry_chunks = [chunk.strip() for chunk in entry_chunks if chunk.strip()]
        
        if not entry_chunks:
            print(f"  ⚠️  No entries found in {year}")
            continue
        
        print(f"  📝 Found {len(entry_chunks)} entries to convert")
        
        # Process each entry
        for j, entry_chunk in enumerate(entry_chunks):
            if not entry_chunk.strip():
                continue
                
            print(f"    🔄 Converting entry {j+1}/{len(entry_chunks)}")
            
            # Parse this single entry
            date_line, items = parse_single_entry(entry_chunk)
            
            if not date_line:
                print(f"      ⚠️  Could not find date line, skipping")
                continue
                
            if not items:
                print(f"      ⚠️  No items found, skipping")
                continue
            
            print(f"      📋 Found {len(items)} items: {date_line}")
            
            # Determine if this is the latest entry (first entry of first year)
            is_latest = (i == 1 and j == 0)
            
            # Organize items while preserving order
            categories = organize_items_preserving_order(items)
            
            # Show categorization summary
            for cat, cat_items in categories.items():
                if cat_items:
                    print(f"        📊 {cat}: {len(cat_items)} items")
            
            # Format new entry
            new_entry = format_new_entry(date_line, categories, is_latest)
            new_content += new_entry
            
            # Add separator between entries (except for last entry)
            if j < len(entry_chunks) - 1:
                new_content += "---\n\n"
        
        # Add separator between years (except for last year)
        if i + 2 < len(year_sections):
            new_content += "---\n\n"
    
    # Write output
    output_path = output_file or input_file.replace('.md', '_converted.md')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"\n✅ Conversion complete!")
    print(f"📄 New format saved to: {output_path}")
    print(f"🔍 Review the file and replace the original if satisfied")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python changelog_converter.py <input_file> [output_file]")
        print("Example: python changelog_converter.py CHANGELOG.md")
        print()
        print("This will convert your changelog to organized format:")
        print("  📄 Frontmatter (open for latest entry)")
        print("  📖 Chapter Updates")  
        print("  🧑‍💻 Lab Updates")
        print("  📚 Appendix")
        print("  (preserving exact original order within each category)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("🔄 Changelog Format Converter")
    print("="*50)
    
    success = convert_changelog(input_file, output_file)
    
    if success:
        print("\n🎉 All done! Items within each entry maintain their exact original order.")
    else:
        print("\n❌ Conversion failed. Check the input file format.")

if __name__ == "__main__":
    main()