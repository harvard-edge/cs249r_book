#!/usr/bin/env python3
"""
Generate changelog entries with exact behavior from original unified script.

This script generates changelog entries by analyzing Git changes since the last
publication, matching the exact behavior of the original changelog-releasenotes.py.
"""

import argparse
import os
import sys
import re
from datetime import datetime
from collections import defaultdict
import yaml

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
CHANGELOG_FILE = "CHANGELOG.md"

# Lab structure from quarto config
LAB_STRUCTURE = None

# Updated to match your actual file structure
chapter_lookup = [
    # MAIN chapters
    ("contents/core/introduction/introduction.qmd", "Introduction", 1),
    ("contents/core/ml_systems/ml_systems.qmd", "ML Systems", 2),
    ("contents/core/dl_primer/dl_primer.qmd", "DL Primer", 3),
    ("contents/core/dnn_architectures/dnn_architectures.qmd", "DNN Architectures", 4),
    ("contents/core/workflow/workflow.qmd", "AI Workflow", 5),
    ("contents/core/data_engineering/data_engineering.qmd", "Data Engineering", 6),
    ("contents/core/frameworks/frameworks.qmd", "AI Frameworks", 7),
    ("contents/core/training/training.qmd", "AI Training", 8),
    ("contents/core/efficient_ai/efficient_ai.qmd", "Efficient AI", 9),
    ("contents/core/optimizations/optimizations.qmd", "Model Optimizations", 10),
    ("contents/core/hw_acceleration/hw_acceleration.qmd", "AI Acceleration", 11),
    ("contents/core/benchmarking/benchmarking.qmd", "Benchmarking AI", 12),
    ("contents/core/ops/ops.qmd", "ML Operations", 13),
    ("contents/core/ondevice_learning/ondevice_learning.qmd", "On-Device Learning", 14),
    ("contents/core/privacy_security/privacy_security.qmd", "Security & Privacy", 15),
    ("contents/core/responsible_ai/responsible_ai.qmd", "Responsible AI", 16),
    ("contents/core/sustainable_ai/sustainable_ai.qmd", "Sustainable AI", 17),
    ("contents/core/robust_ai/robust_ai.qmd", "Robust AI", 18),
    ("contents/core/ai_for_good/ai_for_good.qmd", "AI for Good", 19),
    ("contents/core/conclusion/conclusion.qmd", "Conclusion", 20),
    
    # LAB sections
    ("contents/labs/overview.qmd", "Labs Overview", 100),
    ("contents/labs/getting_started.qmd", "Lab Setup", 101),
    
    # Arduino Nicla Vision Labs
    ("contents/labs/arduino/nicla_vision/setup/setup.qmd", "Arduino Setup", 102),
    ("contents/labs/arduino/nicla_vision/image_classification/image_classification.qmd", "Arduino Image Classification", 103),
    ("contents/labs/arduino/nicla_vision/object_detection/object_detection.qmd", "Arduino Object Detection", 104),
    ("contents/labs/arduino/nicla_vision/kws/kws.qmd", "Arduino Keyword Spotting", 105),
    ("contents/labs/arduino/nicla_vision/motion_classification/motion_classification.qmd", "Arduino Motion Classification", 106),
    
    # Seeed XIAO ESP32S3 Labs
    ("contents/labs/seeed/xiao_esp32s3/setup/setup.qmd", "XIAO Setup", 107),
    ("contents/labs/seeed/xiao_esp32s3/image_classification/image_classification.qmd", "XIAO Image Classification", 108),
    ("contents/labs/seeed/xiao_esp32s3/object_detection/object_detection.qmd", "XIAO Object Detection", 109),
    ("contents/labs/seeed/xiao_esp32s3/kws/kws.qmd", "XIAO Keyword Spotting", 110),
    ("contents/labs/seeed/xiao_esp32s3/motion_classification/motion_classification.qmd", "XIAO Motion Classification", 111),
    
    # Raspberry Pi Labs
    ("contents/labs/raspi/setup/setup.qmd", "Raspberry Pi Setup", 112),
    ("contents/labs/raspi/image_classification/image_classification.qmd", "Pi Image Classification", 113),
    ("contents/labs/raspi/object_detection/object_detection.qmd", "Pi Object Detection", 114),
    ("contents/labs/raspi/llm/llm.qmd", "Pi Large Language Models", 115),
    ("contents/labs/raspi/vlm/vlm.qmd", "Pi Vision Language Models", 116),
    
    # Frontmatter
    ("contents/frontmatter/foreword.qmd", "Foreword", 200),
    ("contents/frontmatter/about/about.qmd", "About", 201),
    ("contents/frontmatter/changelog/changelog.qmd", "Changelog", 202),
    ("contents/frontmatter/acknowledgements/acknowledgements.qmd", "Acknowledgements", 203),
    ("contents/frontmatter/socratiq/socratiq.qmd", "SocratiQ", 204),
    
    # Appendix
    ("contents/appendix/phd_survival_guide.qmd", "PhD Survival Guide", 300),
]

chapter_order = []

def load_lab_structure(quarto_file="book/config/_quarto-html.yml"):
    """Load lab structure from quarto HTML config file."""
    global LAB_STRUCTURE
    
    if not os.path.exists(quarto_file):
        print(f"⚠️ Quarto config file not found: {quarto_file}")
        return None
    
    try:
        with open(quarto_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Extract lab sections from the sidebar structure
        lab_sections = {}
        
        if 'website' in config and 'sidebar' in config['website']:
            for i, section in enumerate(config['website']['sidebar']):
                if isinstance(section, dict):
                    # Look for lab-related sections
                    section_id = section.get('id', '')
                    section_title = section.get('section', '')
                    
                    # Check if this is a lab section or contains lab sections
                    if any(keyword in section_id.lower() for keyword in ['arduino', 'seeed', 'grove', 'raspberry', 'shared', 'labs']):
                        lab_sections[section_title] = []
                        
                        # Extract file paths from contents
                        if 'contents' in section:
                            for item in section['contents']:
                                if isinstance(item, dict) and 'href' in item:
                                    file_path = item['href']
                                    # Convert to the actual file path format used in git
                                    if file_path.startswith('contents/'):
                                        file_path = f"book/{file_path}"
                                    lab_sections[section_title].append(file_path)
                    
                    # Also check if this section contains nested lab sections
                    elif 'contents' in section:
                        for item in section['contents']:
                            if isinstance(item, dict):
                                item_id = item.get('id', '')
                                item_title = item.get('section', '')
                                
                                # Check if this nested item is a lab section
                                if any(keyword in item_id.lower() for keyword in ['arduino', 'seeed', 'grove', 'raspberry', 'shared', 'labs']):
                                    lab_sections[item_title] = []
                                    
                                    # Extract file paths from nested contents
                                    if 'contents' in item:
                                        for nested_item in item['contents']:
                                            if isinstance(nested_item, dict) and 'href' in nested_item:
                                                file_path = nested_item['href']
                                                # Convert to the actual file path format used in git
                                                if file_path.startswith('contents/'):
                                                    file_path = f"book/{file_path}"
                                                lab_sections[item_title].append(file_path)
        
        LAB_STRUCTURE = lab_sections
        print(f"✅ Loaded lab structure with {len(lab_sections)} groups")
        return lab_sections
        
    except Exception as e:
        print(f"❌ Error loading lab structure: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_lab_group_for_file(file_path):
    """Determine which lab group a file belongs to based on the structure."""
    if not LAB_STRUCTURE:
        return None
    
    # Normalize the file path for comparison
    normalized_path = file_path.replace('book/', '')
    
    for group_name, files in LAB_STRUCTURE.items():
        for group_file in files:
            # Convert group file path to normalized format
            group_file_normalized = group_file.replace('book/', '')
            
            # Check if the file matches this group
            if normalized_path == group_file_normalized:
                return group_name
    
    return None

def organize_labs_by_structure(lab_entries):
    """Organize lab entries according to the structure from quarto config."""
    if not LAB_STRUCTURE:
        # Fallback to flat list if no structure loaded
        return lab_entries
    
    # Group lab entries by their hardware platform
    lab_groups = defaultdict(list)
    
    for entry in lab_entries:
        # Extract file path from the entry (assuming format: "**Title**: Updated content...")
        # We need to match this with the actual file paths
        # For now, we'll use a simple heuristic based on the title
        if "Arduino" in entry or "nicla" in entry.lower():
            lab_groups["Arduino"].append(entry)
        elif "Seeed" in entry or "xiao" in entry.lower():
            lab_groups["Seeed XIAO ESP32S3"].append(entry)
        elif "Grove" in entry or "grove" in entry.lower():
            lab_groups["Grove Vision"].append(entry)
        elif "Raspberry" in entry or "raspi" in entry.lower() or "pi " in entry.lower():
            lab_groups["Raspberry Pi"].append(entry)
        elif "Shared" in entry or "shared" in entry.lower() or "kws_feature" in entry.lower() or "dsp_spectral" in entry.lower():
            lab_groups["Shared"].append(entry)
        elif "Hands-on" in entry or "labs" in entry.lower():
            lab_groups["Hands-on Labs"].append(entry)
        else:
            # Default to a general labs group
            lab_groups["Other Labs"].append(entry)
    
    # Sort each group by impact level and build the organized output
    organized_labs = []
    
    # Use the order from the quarto config
    for group_name in LAB_STRUCTURE.keys():
        if group_name in lab_groups:
            sorted_entries = sort_by_impact_level(lab_groups[group_name])
            if sorted_entries:
                # Calculate total changes for this group
                total_changes = sum(int(re.search(r'(\d+) changes', entry).group(1)) 
                                  for entry in sorted_entries 
                                  if re.search(r'(\d+) changes', entry))
                
                organized_labs.append(f"- **{group_name}**: Updated content with {total_changes} changes")
                for entry in sorted_entries:
                    # Extract just the title and changes, remove the group prefix
                    title_match = re.search(r'\*\*(.*?)\*\*: Updated content with (\d+) changes', entry)
                    if title_match:
                        title = title_match.group(1)
                        changes = title_match.group(2)
                        organized_labs.append(f"  - {title}: Updated content with {changes} changes")
    
    return organized_labs

def load_chapter_order(quarto_file=None):
    """Load chapter order from quarto config file."""
    global chapter_order
    
    if not quarto_file:
        quarto_file = "book/config/_quarto-pdf.yml"
    
    if not os.path.exists(quarto_file):
        print(f"⚠️ Quarto config file not found: {quarto_file}")
        return
    
    try:
        with open(quarto_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        def find_chapters(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "chapters":
                        return value
                    result = find_chapters(value)
                    if result:
                        return result
            elif isinstance(obj, list):
                for item in obj:
                    result = find_chapters(item)
                    if result:
                        return result
            return None

        def extract_qmd_paths(items):
            paths = []
            for item in items:
                if isinstance(item, str) and item.endswith(".qmd"):
                    paths.append(item)
                elif isinstance(item, dict):
                    if "chapters" in item:
                        paths.extend(extract_qmd_paths(item["chapters"]))
                    elif "part" in item and isinstance(item["part"], str):
                        if item["part"].endswith(".qmd"):
                            paths.append(item["part"])
                        if "chapters" in item:
                            paths.extend(extract_qmd_paths(item["chapters"]))
            return paths

        chapters_section = find_chapters(config)
        chapter_order = extract_qmd_paths(chapters_section) if chapters_section else []
        
        print(f"📚 Loaded {len(chapter_order)} chapters from {quarto_file}")
        
    except Exception as e:
        print(f"❌ Error loading chapter order: {e}")
        chapter_order = []

def run_git_command(cmd, verbose=False, retries=3):
    """Run a git command and return the output."""
    import subprocess
    
    for attempt in range(retries):
        if verbose:
            print(f"📦 Running: {' '.join(cmd)} (attempt {attempt + 1})")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        
        if attempt < retries - 1:
            print(f"⚠️ Git command failed, retrying in 2s: {result.stderr}")
            import time
            time.sleep(2)
        else:
            raise RuntimeError(f"Git command failed after {retries} attempts: {' '.join(cmd)}\n{result.stderr}")

def extract_chapter_title(file_path):
    """Extract chapter title from file path using lookup table."""
    # Try exact path match first
    for fname, title, number in chapter_lookup:
        if fname == file_path:
            if number <= 20:
                return f"Chapter {number}: {title}"
            elif number <= 199:
                return f"Lab: {title}"
            elif number <= 299:
                return title  # Frontmatter - just use title
            else:
                return title  # Appendix - just use title
    
    # Fallback: try basename matching for backwards compatibility
    base = os.path.basename(file_path)
    for fname, title, number in chapter_lookup:
        if os.path.basename(fname) == base:
            if number <= 20:
                return f"Chapter {number}: {title}"
            elif number <= 199:
                return f"Lab: {title}"
            elif number <= 299:
                return title
            else:
                return title
    
    # Final fallback: generate from path
    if "contents/core/" in file_path:
        return f"Chapter: {base.replace('_', ' ').replace('.qmd', '').title()}"
    elif "contents/labs/" in file_path:
        return f"Lab: {base.replace('_', ' ').replace('.qmd', '').title()}"
    elif "contents/frontmatter/" in file_path:
        return base.replace('_', ' ').replace('.qmd', '').title()
    elif "contents/appendix/" in file_path:
        return base.replace('_', ' ').replace('.qmd', '').title()
    else:
        return base.replace('_', ' ').replace('.qmd', '').title()

def sort_by_impact_level(updates):
    """Sort updates by impact level (number of changes)."""
    def extract_impact_level(update):
        # Extract impact bars from the start of each update
        match = re.search(r'(\d+) changes', update)
        return int(match.group(1)) if match else 0
    
    return sorted(updates, key=extract_impact_level, reverse=True)

def get_changes_in_dev_since(date_start, date_end=None, verbose=False):
    """Get all changes in dev branch since a specific date."""
    cmd = ["git", "log", "--numstat", "--since", date_start]
    if date_end:
        cmd += ["--until", date_end]
    cmd += ["origin/dev", "--", "contents/**/*.qmd"]
    return run_git_command(cmd, verbose=verbose)

def get_commit_messages_for_file(file_path, since, until=None, verbose=False):
    """Get commit messages for a specific file since a date."""
    cmd = ["git", "log", "--pretty=format:%s", "--since", since]
    if until:
        cmd += ["--until", until]
    cmd += ["origin/dev", "--", file_path]
    messages = run_git_command(cmd, verbose=verbose)
    
    # Return all commit messages - let AI determine importance
    meaningful_messages = []
    for message in messages.splitlines():
        if message.strip():
            meaningful_messages.append(message.strip())
    
    return "\n".join(meaningful_messages)

def format_friendly_date(date_str):
    """Format date string to friendly format."""
    try:
        # Try ISO format first (with T separator)
        if 'T' in date_str:
            dt = datetime.fromisoformat(date_str)
        else:
            # Fallback to space-separated format
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z")
        # Format as "January 28 at 02:36 PM" (full month name)
        return dt.strftime("%B %d at %I:%M %p")
    except:
        return date_str

def normalized_path(path):
    """Normalize path for comparison."""
    return os.path.normpath(path).lower()

def generate_entry(start_date, end_date=None, verbose=False, is_latest=False):
    """Generate a changelog entry for the specified time period."""
    if verbose:
        print(f"📁 Processing changes from {start_date} to {end_date or 'now'}")
    print(f"🔍 Analyzing Git changes...")
    changes = get_changes_in_dev_since(start_date, end_date, verbose=verbose)
    if not changes.strip():
        print("  ⚠️ No changes found in specified period")
        return None

    print("📊 Categorizing changes by file...")
    changes_by_file = defaultdict(lambda: [0, 0])
    for line in changes.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added, removed, file_path = parts
        added = int(added) if added.isdigit() else 0
        removed = int(removed) if removed.isdigit() else 0
        changes_by_file[file_path][0] += added
        changes_by_file[file_path][1] += removed

    current_date = datetime.now().strftime('%B %d at %I:%M %p') if not end_date else format_friendly_date(end_date)
    entry = f"### 📅 {current_date}\n\n"

    frontmatter, chapters, labs, appendix = [], [], [], []

    ordered_files = sorted(
        changes_by_file,
        key=lambda f: next(
            (i for i, ch in enumerate(chapter_order) if normalized_path(f).endswith(normalized_path(ch))),
            float('inf')
        )
    )

    total_files = len(ordered_files)
    print(f"📝 Processing {total_files} changed files...")
    
    for idx, file_path in enumerate(ordered_files, 1):
        added, removed = changes_by_file[file_path]
        total = added + removed
        if verbose:
            print(f"🔍 Summarizing {file_path} ({added}+ / {removed}-) [{idx}/{total_files}]")
        else:
            print(f"  📄 [{idx}/{total_files}] {os.path.basename(file_path)} ({added}+ {removed}-)")
        
        # Skip references
        if "references.qmd" in file_path:
            continue
            
        commit_msgs = get_commit_messages_for_file(file_path, start_date, end_date, verbose=verbose)
        
        # Skip if no meaningful commits
        if not commit_msgs.strip():
            if verbose:
                print(f"⏭️ Skipping {file_path} - no meaningful changes")
            continue
            
        print(f"    📝 Generating summary...")
        # Create simple summary based on file path and commit count
        chapter_title = extract_chapter_title(file_path)
        commit_count = len([msg for msg in commit_msgs.split('\n') if msg.strip()])
        summary = f"- **{chapter_title}**: Updated content with {commit_count} changes"
        
        # Show the generated summary
        summary_text = summary.replace(f"- **{chapter_title}**: ", "")
        print(f"      📝 {summary_text}")
        
        # Categorize by content type
        if "contents/frontmatter/" in file_path:
            frontmatter.append(summary)
        elif "contents/labs/" in file_path:
            labs.append(summary)
        elif "contents/appendix/" in file_path:
            appendix.append(summary)
        else:
            chapters.append(summary)

    print(f"📋 Organizing into sections...")
    print(f"  📄 Frontmatter: {len(frontmatter)} entries")
    print(f"  📖 Chapters: {len(chapters)} entries")
    print(f"  🧑‍💻 Labs: {len(labs)} entries")
    print(f"  📚 Appendix: {len(appendix)} entries")

    # Determine if sections should be open or closed
    # All entries should be closed by default - let users choose what to explore
    details_state = ""  # Always closed for better UX

    # Add sections in order: Frontmatter, Chapters, Labs, Appendix
    if frontmatter:
        entry += f"<details {details_state}>\n<summary>**📄 Frontmatter**</summary>\n\n" + "\n".join(sort_by_impact_level(frontmatter)) + "\n\n</details>\n\n"
    if chapters:
        entry += f"<details {details_state}>\n<summary>**📖 Chapters**</summary>\n\n" + "\n".join(sort_by_impact_level(chapters)) + "\n\n</details>\n\n"
    if labs:
        # Organize labs according to the structure from quarto config
        organized_labs = organize_labs_by_structure(labs)
        entry += f"<details {details_state}>\n<summary>**🧑‍💻 Labs**</summary>\n\n" + "\n".join(organized_labs) + "\n\n</details>\n\n"
    if appendix:
        entry += f"<details {details_state}>\n<summary>**📚 Appendix**</summary>\n\n" + "\n".join(sort_by_impact_level(appendix)) + "\n\n</details>\n"

    # If no content sections were added, return None (empty entry)
    if not frontmatter and not chapters and not labs and not appendix:
        print("  ⚠️ No meaningful content changes found - skipping entry")
        return None

    print("✅ Entry generation complete")
    return entry

def generate_changelog(mode="incremental", verbose=False):
    """Generate changelog entries."""
    print("🔄 Starting Git data fetch...")
    print("  📦 Fetching gh-pages branch...")
    run_git_command(["git", "fetch", "origin", "gh-pages:refs/remotes/origin/gh-pages"], verbose=verbose)
    print("  📦 Fetching dev branch...")
    run_git_command(["git", "fetch", "origin", "dev:refs/remotes/origin/dev"], verbose=verbose)
    print("✅ Git data fetch complete")

    def get_latest_gh_pages_commit():
        print("🔍 Looking for latest publication commit...")
        # Look for actual publication commits, not administrative ones
        output = run_git_command(["git", "log", "--pretty=format:%H %aI", "--grep=Built site for gh-pages", "origin/gh-pages"], verbose=verbose)
        if output.strip():
            first_line = output.split('\n')[0]
            parts = first_line.split(" ", 1)
            result = (parts[0], parts[1]) if len(parts) == 2 else (None, None)
            if result[0]:
                print(f"  📅 Found latest commit: {result[0][:8]} from {result[1]}")
            return result
        print("  ⚠️ No publication commits found")
        return (None, None)

    def get_all_gh_pages_commits():
        print("🔍 Scanning all publication commits...")
        # Look for actual publication commits, not administrative ones
        output = run_git_command(["git", "log", "--pretty=format:%H %aI", "--grep=Built site for gh-pages", "origin/gh-pages"], verbose=verbose)
        commits = []
        for line in output.splitlines():
            parts = line.split(" ", 1)
            if len(parts) == 2:
                commits.append((parts[0], parts[1]))
        print(f"  📊 Found {len(commits)} publication commits")
        return commits

    def extract_year_from_date(date_str):
        try:
            # Try ISO format first (2023-09-16T22:16:31-04:00)
            return datetime.fromisoformat(date_str.replace('Z', '+00:00')).year
        except:
            try:
                # Try the old format as fallback
                return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z").year
            except:
                return datetime.now().year

    latest_commit, latest_date = get_latest_gh_pages_commit()

    if mode == "full":
        if verbose:
            print("🔁 Running full regeneration...")
        commits = get_all_gh_pages_commits()
        
        # Group commits by date (YYYY-MM-DD) to merge same-day publishes
        def extract_date_only(date_str):
            try:
                # Try ISO format first (2023-09-16T22:16:31-04:00)
                return datetime.fromisoformat(date_str.replace('Z', '+00:00')).strftime("%Y-%m-%d")
            except:
                try:
                    # Try the old format as fallback
                    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z").strftime("%Y-%m-%d")
                except:
                    return date_str.split()[0]  # fallback to first part
        
        # Group commits by publication date
        commits_by_date = defaultdict(list)
        for commit, date in commits:
            date_key = extract_date_only(date)
            commits_by_date[date_key].append((commit, date))
        
        # Sort dates and get unique publication periods
        unique_dates = sorted(commits_by_date.keys(), reverse=True)  # newest first
        print(f"📊 Found {len(unique_dates)} unique publication dates...")
        
        # Group entries by year
        entries_by_year = defaultdict(list)
        
        for i in range(len(unique_dates) - 1):
            current_date_key = unique_dates[i]
            previous_date_key = unique_dates[i + 1]
            
            # Get the latest commit from current date for the "published on" date
            current_commits = commits_by_date[current_date_key]
            latest_current = max(current_commits, key=lambda x: x[1])  # latest timestamp
            
            # Get the earliest commit from previous date as the "since" date
            previous_commits = commits_by_date[previous_date_key]
            earliest_previous = min(previous_commits, key=lambda x: x[1])  # earliest timestamp
            
            current_date = latest_current[1]
            previous_date = earliest_previous[1]
            
            # Extract year from current_date (the publication date)
            pub_year = extract_year_from_date(current_date)
            
            print(f"📅 Processing period {i+1}/{len(unique_dates)-1}: {format_friendly_date(previous_date)} → {format_friendly_date(current_date)} [{pub_year}]")
            entry = generate_entry(previous_date, current_date, verbose=verbose, is_latest=(i==0))
            if entry:
                entries_by_year[pub_year].append(entry)
        
        if not entries_by_year:
            return "_No updates found._"
        
        # Build output with year headers, newest years first
        output_sections = []
        for year in sorted(entries_by_year.keys(), reverse=True):
            year_header = f"## {year} Updates"
            year_entries = "\n\n".join(entries_by_year[year])
            output_sections.append(f"{year_header}\n\n{year_entries}")
        
        return "\n\n---\n\n".join(output_sections) + "\n"
        
    else:
        if verbose:
            print("⚡ Running update mode...")
        print(f"📅 Processing changes since: {format_friendly_date(latest_date) if latest_date else 'beginning'}")
        entry = generate_entry(latest_date, verbose=verbose, is_latest=True)
        if not entry:
            return "_No updates found._"
        
        # Extract year from the latest date instead of using current year
        if latest_date:
            current_year = extract_year_from_date(latest_date)
        else:
            current_year = datetime.now().year
        year_header = f"## {current_year} Updates"
        return f"{year_header}\n\n{entry}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate changelog entries with exact behavior from original unified script.")
    
    # Changelog mode arguments
    parser.add_argument("--full", action="store_true", help="Regenerate the entire changelog from scratch.")
    parser.add_argument("--incremental", action="store_true", help="Add new entries since last gh-pages publish.")
    
    # General options
    parser.add_argument("-t", "--test", action="store_true", help="Run without writing to file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("-q", "--quarto-config", type=str, help="Path to quarto config file (default: book/config/_quarto-pdf.yml)")

    args = parser.parse_args()
    
    # Require either --full or --incremental to be specified
    if args.full and args.incremental:
        print("❌ Error: Cannot specify both --full and --incremental modes")
        exit(1)
    elif args.full:
        mode = "full"
    elif args.incremental:
        mode = "update"  # Keep internal name as "update" for compatibility
    else:
        print("❌ Error: Must specify either --full or --incremental")
        print("💡 Use --help for usage information")
        exit(1)

    try:
        load_chapter_order(args.quarto_config)
        # Load lab structure from HTML config (not PDF config)
        load_lab_structure("book/config/_quarto-html.yml")
        
        # Print configuration header
        print("=" * 60)
        print("📝 CHANGELOG GENERATION CONFIG")
        print("=" * 60)
        print(f"🎯 Mode: {mode.upper()}")
        print(f"🔧 Test Mode: {'ON' if args.test else 'OFF'}")
        print(f"📢 Verbose: {'ON' if args.verbose else 'OFF'}")
        print(f"📋 Features: Impact bars, importance sorting, specific summaries")
        print("=" * 60)
        print()
        
        print(f"🚀 Starting changelog generation in {mode} mode...")

        new_entry = generate_changelog(mode=mode, verbose=args.verbose)

        if args.test:
            print("🧪 TEST OUTPUT ONLY:\n")
            print(new_entry)
        else:
            existing = ""
            if os.path.exists(CHANGELOG_FILE):
                with open(CHANGELOG_FILE, "r", encoding="utf-8") as f:
                    existing = f.read()

            current_year = datetime.now().year
            year_header = f"## {current_year} Updates"

            # For update mode, insert new entry after the year header
            if mode == "full":
                # For full mode, replace entire content (already includes year headers)
                updated_content = new_entry.strip()
            else:
                # For incremental, insert new entry after year header
                existing_lines = existing.splitlines()
                new_lines = []
                inserted = False
                
                for line in existing_lines:
                    new_lines.append(line)
                    # Insert new entry right after the year header
                    if not inserted and line.strip() == year_header:
                        # Add the new entry (without year header since it's already in the file)
                        new_entry_lines = new_entry.strip().splitlines()
                        # Skip the first line (year header) since we're inserting after existing year header
                        if new_entry_lines and new_entry_lines[0].strip() == year_header:
                            new_entry_lines = new_entry_lines[1:]
                        new_lines.extend(new_entry_lines)
                        new_lines.append("")  # Add blank line
                        inserted = True
                
                if not inserted:
                    # If no year header found, prepend to beginning
                    new_lines = new_entry.strip().splitlines() + [""] + existing_lines
                
                updated_content = "\n".join(new_lines)

            with open(CHANGELOG_FILE, "w", encoding="utf-8") as f:
                f.write(updated_content.strip() + "\n")

            print(f"\n✅ Changelog written to {CHANGELOG_FILE}")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 