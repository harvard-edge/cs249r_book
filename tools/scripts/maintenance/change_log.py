#!/usr/bin/env python3
"""
Generate changelog entries with AI-powered summaries.

This script generates changelog entries by analyzing Git changes since the last
publication. It can generate simple change counts or AI-powered summaries.
"""

import argparse
import os
import sys
import re
from datetime import datetime
from collections import defaultdict
import yaml

# Global variables
chapter_order = []

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
            chapters = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'contents':
                        chapters.extend(extract_qmd_paths(value))
                    else:
                        chapters.extend(find_chapters(value))
            elif isinstance(obj, list):
                for item in obj:
                    chapters.extend(find_chapters(item))
            return chapters
        
        def extract_qmd_paths(items):
            paths = []
            for item in items:
                if isinstance(item, dict):
                    if 'href' in item:
                        href = item['href']
                        if href.endswith('.qmd'):
                            paths.append(href)
                    if 'contents' in item:
                        paths.extend(extract_qmd_paths(item['contents']))
            return paths
        
        chapter_order = find_chapters(config)
        print(f"📚 Loaded {len(chapter_order)} chapters from {quarto_file}")
        
    except Exception as e:
        print(f"❌ Error loading chapter order: {e}")
        chapter_order = []

def call_ollama(prompt, model="gemma2:9b", url="http://localhost:11434"):
    """Call Ollama API to generate AI summaries."""
    try:
        import requests
        import json
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(f"{url}/api/generate", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            print(f"⚠️ Ollama API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ Error calling Ollama: {e}")
        return None

def generate_ai_summary(chapter_title, commit_messages, file_path, verbose=False):
    """Generate AI summary for a file based on commit messages."""
    if not commit_messages.strip():
        return f"Updated content with minor changes"
    
    # Create a prompt for AI summary
    prompt = f"""Based on these Git commit messages for {chapter_title} ({file_path}), generate a brief, informative summary of what was updated. Focus on the most important changes and improvements.

Commit messages:
{commit_messages}

Generate a concise summary (1-2 sentences) that describes the key updates:"""
    
    if verbose:
        print(f"🤖 Generating AI summary for {chapter_title}...")
    
    ai_summary = call_ollama(prompt)
    
    if ai_summary:
        return ai_summary
    else:
        # Fallback to simple summary
        commit_count = len([msg for msg in commit_messages.split('\n') if msg.strip()])
        return f"Updated content with {commit_count} changes"

def run_git_command(cmd, verbose=False, retries=3):
    """Run a git command and return the output."""
    import subprocess
    
    for attempt in range(retries):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if verbose:
                print(f"  🔧 {' '.join(cmd)}")
            return result.stdout
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"  ❌ {' '.join(cmd)} failed: {e}")
            if attempt == retries - 1:
                print(f"❌ Git command failed after {retries} attempts: {' '.join(cmd)}")
                return ""
            continue
    return ""

def extract_chapter_title(file_path):
    """Extract chapter title from file path."""
    # Try exact path match first
    normalized_file_path = normalized_path(file_path)
    for chapter_path in chapter_order:
        if normalized_file_path.endswith(normalized_path(chapter_path)):
            # Extract title from the chapter path
            basename = os.path.basename(chapter_path)
            if basename.endswith('.qmd'):
                title = basename[:-4]  # Remove .qmd extension
            else:
                title = basename
            
            # Convert to title case and handle special cases
            title = title.replace('_', ' ').title()
            
            # Add chapter number if available
            for i, ch in enumerate(chapter_order, 1):
                if normalized_file_path.endswith(normalized_path(ch)):
                    if "introduction" in title.lower():
                        return f"Chapter 1: Introduction"
                    elif "ml_systems" in title.lower():
                        return f"Chapter 2: ML Systems"
                    elif "dl_primer" in title.lower():
                        return f"Chapter 3: DL Primer"
                    elif "dnn_architectures" in title.lower():
                        return f"Chapter 4: DNN Architectures"
                    elif "workflow" in title.lower():
                        return f"Chapter 5: AI Workflow"
                    elif "data_engineering" in title.lower():
                        return f"Chapter 6: Data Engineering"
                    elif "frameworks" in title.lower():
                        return f"Chapter 7: AI Frameworks"
                    elif "training" in title.lower():
                        return f"Chapter 8: AI Training"
                    elif "efficient_ai" in title.lower():
                        return f"Chapter 9: Efficient AI"
                    elif "optimizations" in title.lower():
                        return f"Chapter 10: Model Optimizations"
                    elif "hw_acceleration" in title.lower():
                        return f"Chapter 11: AI Acceleration"
                    elif "benchmarking" in title.lower():
                        return f"Chapter 12: Benchmarking AI"
                    elif "ops" in title.lower():
                        return f"Chapter 13: ML Operations"
                    elif "ondevice_learning" in title.lower():
                        return f"Chapter 14: On-Device Learning"
                    elif "privacy_security" in title.lower():
                        return f"Chapter 15: Security & Privacy"
                    elif "responsible_ai" in title.lower():
                        return f"Chapter 16: Responsible AI"
                    elif "sustainable_ai" in title.lower():
                        return f"Chapter 17: Sustainable AI"
                    elif "robust_ai" in title.lower():
                        return f"Chapter 18: Robust AI"
                    elif "ai_for_good" in title.lower():
                        return f"Chapter 19: AI for Good"
                    elif "conclusion" in title.lower():
                        return f"Chapter 20: Conclusion"
                    else:
                        return f"Chapter {i}: {title}"
    
    # Fallback: extract from file path
    basename = os.path.basename(file_path)
    if basename.endswith('.qmd'):
        title = basename[:-4]
    else:
        title = basename
    
    return title.replace('_', ' ').title()

def sort_by_impact_level(updates):
    """Sort updates by impact level (number of changes)."""
    def extract_impact_level(update):
        # Extract impact bars from the start of each update
        match = re.search(r'(\d+) changes', update)
        return int(match.group(1)) if match else 0
    
    return sorted(updates, key=extract_impact_level, reverse=True)

def get_changes_in_dev_since(date_start, date_end=None, verbose=False):
    """Get all changes in dev branch since a specific date."""
    cmd = ["git", "log", "--pretty=format:", "--numstat", f"--since={date_start}"]
    if date_end:
        cmd.append(f"--until={date_end}")
    cmd.extend(["origin/dev", "--"])
    return run_git_command(cmd, verbose=verbose)

def get_commit_messages_for_file(file_path, since, until=None, verbose=False):
    """Get commit messages for a specific file since a date."""
    cmd = ["git", "log", "--pretty=format:%s", f"--since={since}"]
    if until:
        cmd.append(f"--until={until}")
    cmd.extend(["origin/dev", "--", file_path])
    return run_git_command(cmd, verbose=verbose)

def format_friendly_date(date_str):
    """Format date string to friendly format."""
    try:
        # Try ISO format first (2023-09-16T22:16:31-04:00)
        if 'T' in date_str:
            dt = datetime.fromisoformat(date_str)
        else:
            # Fallback to space-separated format
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z")
        return dt.strftime("%B %d at %I:%M %p")
    except:
        return date_str

def normalized_path(path):
    """Normalize path for comparison."""
    return os.path.normpath(path).lower()

def generate_entry(start_date, end_date=None, verbose=False, is_latest=False, ai_mode=False, ollama_url="http://localhost:11434", ollama_model="gemma2:9b"):
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
    
    # Filter for only content files (qmd files in book content directories)
    content_files = []
    for file_path in ordered_files:
        # Only include .qmd files in book content directories
        if (file_path.endswith('.qmd') and 
            ('book/contents/' in file_path or 
             'contents/' in file_path or
             file_path.startswith('contents/'))):
            content_files.append(file_path)
    
    total_files = len(content_files)
    print(f"📝 Processing {total_files} content files (filtered from {len(ordered_files)} total files)...")
    
    for idx, file_path in enumerate(content_files, 1):
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
        
        # Generate summary based on AI mode
        chapter_title = extract_chapter_title(file_path)
        if ai_mode:
            summary_text = generate_ai_summary(chapter_title, commit_msgs, file_path, verbose=verbose)
            summary = f"- **{chapter_title}**: {summary_text}"
        else:
            # Create simple summary based on file path and commit count
            commit_count = len([msg for msg in commit_msgs.split('\n') if msg.strip()])
            summary_text = f"Updated content with {commit_count} changes"
            summary = f"- **{chapter_title}**: {summary_text}"
        
        # Show the generated summary
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
    details_state = ""  # Always closed for better UX

    # Add sections in order: Frontmatter, Chapters, Labs, Appendix
    if frontmatter:
        entry += f"<details {details_state}>\n<summary>**📄 Frontmatter**</summary>\n\n" + "\n".join(sort_by_impact_level(frontmatter)) + "\n\n</details>\n\n"
    if chapters:
        entry += f"<details {details_state}>\n<summary>**📖 Chapters**</summary>\n\n" + "\n".join(sort_by_impact_level(chapters)) + "\n\n</details>\n\n"
    if labs:
        entry += f"<details {details_state}>\n<summary>**🧑‍💻 Labs**</summary>\n\n" + "\n".join(sort_by_impact_level(labs)) + "\n\n</details>\n\n"
    if appendix:
        entry += f"<details {details_state}>\n<summary>**📚 Appendix**</summary>\n\n" + "\n".join(sort_by_impact_level(appendix)) + "\n\n</details>\n"

    # If no content sections were added, return None (empty entry)
    if not frontmatter and not chapters and not labs and not appendix:
        print("  ⚠️ No meaningful content changes found - skipping entry")
        return None

    print("✅ Entry generation complete")
    return entry

def generate_changelog(mode="incremental", verbose=False, ai_mode=False, ollama_url="http://localhost:11434", ollama_model="gemma2:9b"):
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
        
        for date_key in unique_dates:
            # Get the latest commit for this date
            latest_commit_for_date = commits_by_date[date_key][0]
            entry = generate_entry(latest_commit_for_date[1], verbose=verbose, ai_mode=ai_mode, ollama_url=ollama_url, ollama_model=ollama_model)
            if entry:
                year = extract_year_from_date(latest_commit_for_date[1])
                entries_by_year[year].append(entry)
        
        # Build output with year headers
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
        entry = generate_entry(latest_date, verbose=verbose, is_latest=True, ai_mode=ai_mode, ollama_url=ollama_url, ollama_model=ollama_model)
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
    parser = argparse.ArgumentParser(description="Generate changelog entries with AI-powered summaries.")
    
    # Changelog mode arguments
    parser.add_argument("--full", action="store_true", help="Regenerate the entire changelog from scratch.")
    parser.add_argument("--incremental", action="store_true", help="Add new entries since last gh-pages publish.")
    
    # General options
    parser.add_argument("-t", "--test", action="store_true", help="Run without writing to file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("-q", "--quarto-config", type=str, help="Path to quarto config file (default: book/config/_quarto-pdf.yml)")
    
    # AI options
    parser.add_argument("--ai-mode", action="store_true", help="Enable AI-generated summaries instead of simple change counts.")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL for AI summaries.")
    parser.add_argument("--ollama-model", default="gemma2:9b", help="Ollama model to use for AI summaries.")

    args = parser.parse_args()
    
    # Load configuration
    try:
        load_chapter_order(args.quarto_config)
        
        # Print configuration header
        print("=" * 60)
        print("📝 CHANGELOG GENERATION CONFIG")
        print("=" * 60)
        print(f"🎯 Mode: {'FULL' if args.full else 'UPDATE'}")
        print(f"🔧 Test Mode: {'ON' if args.test else 'OFF'}")
        print(f"📢 Verbose: {'ON' if args.verbose else 'OFF'}")
        print(f"🤖 AI Mode: {'ON' if args.ai_mode else 'OFF'}")
        if args.ai_mode:
            print(f"🤖 AI Model: {args.ollama_model}")
            print(f"🤖 AI URL: {args.ollama_url}")
        print(f"📋 Features: Impact bars, importance sorting, specific summaries")
        print("=" * 60)
        print()
        
        print("🚀 Starting changelog generation...")
        
        # Determine mode
        mode = "full" if args.full else "incremental"
        
        # Generate changelog
        new_entry = generate_changelog(mode=mode, verbose=args.verbose, ai_mode=args.ai_mode, ollama_url=args.ollama_url, ollama_model=args.ollama_model)
        
        if args.test:
            # Display the generated content for test mode
            print("🧪 TEST MODE - Generated changelog entry:")
            print("=" * 60)
            print(new_entry)
            print("=" * 60)
            print(f"📊 Content length: {len(new_entry)} characters")
        else:
            # Write to CHANGELOG.md
            if new_entry and new_entry != "_No updates found._":
                # Read existing changelog
                changelog_file = "CHANGELOG.md"
                existing_content = ""
                if os.path.exists(changelog_file):
                    with open(changelog_file, 'r', encoding='utf-8') as f:
                        existing_content = f.read()
                
                # Insert new entry at the top
                if existing_content.strip():
                    updated_content = new_entry + "\n\n---\n\n" + existing_content
                else:
                    updated_content = new_entry
                
                # Write back to file
                with open(changelog_file, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                
                print(f"✅ Changelog updated: {changelog_file}")
                print(f"📊 File size: {len(updated_content)} characters")
            else:
                print("ℹ️ No changes to add to changelog")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 