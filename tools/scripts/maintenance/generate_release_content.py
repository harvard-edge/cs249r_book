#!/usr/bin/env python3
"""
Generate changelog entries and release notes using AI analysis.

This script analyzes git commits to generate:
1. Changelog entries for the CHANGELOG.md file
2. Release notes for GitHub releases

Features:
- AI-powered commit analysis using Ollama
- Categorization of changes (features, fixes, docs, etc.)
- Impact assessment and importance ranking
- Customizable AI models
- Support for both changelog and release notes modes

Usage:
  # Generate changelog entry
  python generate_release_content.py --changelog

  # Generate release notes
  python generate_release_content.py --release-notes --version v1.2.0 --previous-version v1.1.0 --description "New features"
"""

import subprocess
import re
import os
import argparse
import yaml
import time
import requests
import json
from collections import defaultdict
from datetime import datetime
# Initialize Ollama as default
use_ollama = True  # Global flag to track which service to use

CHANGELOG_FILE = "CHANGELOG.md"
QUARTO_YML_FILE = "book/config/_quarto-pdf.yml"  # Default to PDF config which has chapters structure
GITHUB_REPO_URL = "https://github.com/harvard-edge/cs249r_book/"
# Removed MAJOR_CHANGE_THRESHOLD since we're organizing by content type now
OPENAI_DELAY = 1  # seconds between API calls
OLLAMA_DELAY = 0.5  # seconds between Ollama calls (faster since local)
OLLAMA_URL = "http://localhost:11434/api/generate"  # Default Ollama API endpoint

chapter_order = []

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

def load_chapter_order(quarto_file=None):
    global chapter_order
    config_file = quarto_file or QUARTO_YML_FILE
    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

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

    chapters_section = find_chapters(data)
    chapter_order = extract_qmd_paths(chapters_section) if chapters_section else []

    print(f"📚 Loaded {len(chapter_order)} chapters from {config_file}")

def run_git_command(cmd, verbose=False, retries=3):
    for attempt in range(retries):
        if verbose:
            print(f"📦 Running: {' '.join(cmd)} (attempt {attempt + 1})")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        
        if attempt < retries - 1:
            print(f"⚠️ Git command failed, retrying in 2s: {result.stderr}")
            time.sleep(2)
        else:
            raise RuntimeError(f"Git command failed after {retries} attempts: {' '.join(cmd)}\n{result.stderr}")

def extract_chapter_title(file_path):
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
    def extract_impact_level(update):
        # Extract impact bars from the start of each update
        import re
        match = re.search(r'`([█░]+)`', update)
        if match:
            bars = match.group(1)
            # Count filled bars (█) - higher count = higher importance
            filled_count = bars.count('█')
            return -filled_count  # Negative for descending order (most important first)
        return 0  # Default for entries without impact bars
    return sorted(updates, key=extract_impact_level)

def get_changes_in_dev_since(date_start, date_end=None, verbose=False):
    cmd = ["git", "log", "--numstat", "--since", date_start]
    if date_end:
        cmd += ["--until", date_end]
    cmd += ["origin/dev", "--", "contents/**/*.qmd"]
    return run_git_command(cmd, verbose=verbose)



def get_commit_messages_for_file(file_path, since, until=None, verbose=False):
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

def call_ollama(prompt, model="llama3.1:8b", verbose=False):
    """Call Ollama API for text generation."""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 100
            }
        }
        
        if verbose:
            print(f"🤖 Calling Ollama with model: {model}")
            
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "").strip()
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Ollama API error: {e}")
        return None
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
        return None

def summarize_changes_with_openai(file_path, commit_messages, verbose=False, max_retries=3, use_ollama=False, ollama_model="llama3.1:8b"):
    chapter_title = extract_chapter_title(file_path)
    if verbose:
        print(f"🤖 Calling {'Ollama' if use_ollama else 'OpenAI'} for: {file_path} -- {chapter_title}")

    prompt = f"""You're writing a changelog entry for a machine learning textbook. Readers and instructors need to know what changed and how important it is.

File: {file_path}
Chapter: {chapter_title}

Commit messages:
{commit_messages}

The output format will be: `[IMPACT]` **{chapter_title}**: [YOUR SUMMARY]

Since the chapter title is already shown, DO NOT repeat it in your summary. Just state what changed directly.

First, analyze the commits and list the main changes. Then write ONE specific sentence about what changed.
Finally, rate the importance (1-5 bars). Be realistic about impact - most changes should be Small or Medium:
- █████ Major: New chapters, sections, or significant rewrites (rare)
- ████░ Large: Multiple examples, new concepts, substantial updates (uncommon)
- ███░░ Medium: New examples, clarifications, moderate changes (common)
- ██░░░ Small: Minor fixes, formatting, small corrections, single example additions (most common)
- █░░░░ Tiny: Typos, punctuation, very minor tweaks (use this more often)

Format your response exactly like this:
CHANGES: [list 2-3 main changes from commits]
SUMMARY: [what changed - NO chapter name, just the changes]
IMPACT: [█████, ████░, ███░░, ██░░░, or █░░░░]

Example:
CHANGES: Added transformer architecture section, New attention mechanism diagrams, Fixed backprop equations
SUMMARY: Added transformer architecture section with attention mechanism diagrams and corrected backpropagation equations
IMPACT: ████░

BAD: "Chapter 10 has been updated with new content"
GOOD: "Added transformer architecture section with attention mechanism diagrams"

GOOD examples:
- "Added lottery ticket hypothesis section with pruning examples"
- "New GPU memory optimization diagrams and CUDA code samples" 
- "Expanded federated learning coverage with privacy-preserving techniques"
- "Fixed mathematical notation in backpropagation equations"

BAD examples (don't use):
- "The chapter has been revised to include..."
- "Updated with new sections and examples..."
- "Enhanced with improved clarity and..."
- "Modified to add new content about..."

Focus on WHAT was added/changed, not HOW it was changed. Use varied sentence structures.
Return only the description (no chapter title, no bullet points)."""

    for attempt in range(max_retries):
        try:
            # Add delay only for OpenAI (rate limiting)
            if not use_ollama and attempt > 0:
                time.sleep(OPENAI_DELAY * (2 ** attempt))  # exponential backoff
            
            if use_ollama:
                summary = call_ollama(prompt, model=ollama_model, verbose=verbose)
                if summary is None:
                    raise Exception("Ollama call failed")
            else:
                response = client.chat.completions.create(
                    model="gpt-4",
                    temperature=0.2,  # Lower temperature for more consistent output
                    max_tokens=100,   # Limit length for conciseness
                    messages=[
                        {"role": "system", "content": "You are a technical writer creating concise changelog entries. Be specific and avoid generic language."},
                        {"role": "user", "content": prompt}
                    ]
                )
                summary = response.choices[0].message.content.strip()

            if not summary:
                return f"- **{chapter_title}**: _(no meaningful changes detected)_"

            # Parse the new structured format
            import re
            summary_match = re.search(r'SUMMARY:\s*(.+?)(?:\nIMPACT:|$)', summary, re.DOTALL)
            impact_match = re.search(r'IMPACT:\s*([█░]+)', summary)
            
            if summary_match:
                parsed_summary = summary_match.group(1).strip()
                # Remove any trailing punctuation
                if parsed_summary.endswith("."):
                    parsed_summary = parsed_summary[:-1]
                
                impact_bars = impact_match.group(1) if impact_match else "███░░"  # default medium
                
                # Add delay only for OpenAI after successful call
                if not use_ollama:
                    time.sleep(OPENAI_DELAY)
                return f"- `{impact_bars}` **{chapter_title}**: {parsed_summary}"
            else:
                # Fallback to old format if parsing fails
                summary = summary.replace("--- --- --- ---", "").strip()
                if summary.endswith("."):
                    summary = summary[:-1]
                
                if not use_ollama:
                    time.sleep(OPENAI_DELAY)
                return f"- **{chapter_title}**: {summary}"

        except Exception as e:
            print(f"⚠️ {'Ollama' if use_ollama else 'OpenAI'} attempt {attempt + 1} failed for {file_path}: {e}")
            if attempt == max_retries - 1:
                return f"- **{chapter_title}**: _(unable to summarize; see commits manually)_"

def format_friendly_date(date_str):
    try:
        # Try ISO format first (with T separator)
        if 'T' in date_str:
            dt = datetime.fromisoformat(date_str)
        else:
            # Fallback to space-separated format
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z")
        # Format as "August 01 at 04:54 PM" (full month name, include time)
        return dt.strftime("%B %d at %I:%M %p")
    except:
        return date_str

def normalized_path(path):
    return os.path.normpath(path).lower()

def generate_entry(start_date, end_date=None, verbose=False, is_latest=False):
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
            
        print(f"    🤖 Generating summary...")
        summary = summarize_changes_with_openai(file_path, commit_msgs, verbose=verbose, use_ollama=use_ollama, ollama_model=args.model)
        
        # Show the generated summary
        summary_text = summary.replace(f"- **{extract_chapter_title(file_path)}**: ", "")
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
        entry += f"<details {details_state}>\n<summary>**🧑‍💻 Labs**</summary>\n\n" + "\n".join(sort_by_impact_level(labs)) + "\n\n</details>\n\n"
    if appendix:
        entry += f"<details {details_state}>\n<summary>**📚 Appendix**</summary>\n\n" + "\n".join(sort_by_impact_level(appendix)) + "\n\n</details>\n"

    # If no content sections were added, return None (empty entry)
    if not frontmatter and not chapters and not labs and not appendix:
        print("  ⚠️ No meaningful content changes found - skipping entry")
        return None

    print("✅ Entry generation complete")
    return entry

def generate_demo_entry():
    """Generate a demo changelog entry with real data from the repository."""
    current_date = datetime.now().strftime('%B %d at %I:%M %p')
    current_year = datetime.now().year
    
    # Get some real file paths from the repository
    real_files = [
        "book/contents/frontmatter/about/about.qmd",
        "book/contents/frontmatter/acknowledgements/acknowledgements.qmd",
        "book/contents/core/dl_primer/dl_primer.qmd",
        "book/contents/core/workflow/workflow.qmd",
        "book/contents/core/training/training.qmd",
        "book/contents/core/introduction/introduction.qmd",
        "book/contents/core/benchmarking/benchmarking.qmd",
        "book/contents/labs/arduino/nicla_vision/image_classification/image_classification.qmd",
        "book/contents/labs/raspi/setup/setup.qmd",
        "book/contents/backmatter/resources/phd_survival_guide.qmd"
    ]
    
    # Try to get some real commit data for more realistic content
    try:
        # Get recent commit messages for some files
        recent_commits = run_git_command(["git", "log", "--oneline", "-5", "--", "book/contents/core/dl_primer/dl_primer.qmd"], verbose=False)
        if recent_commits:
            # Use real commit data if available
            pass
    except:
        pass
    
    # Generate realistic summaries based on actual files
    frontmatter_entries = [
        "**About**: Updated book description and target audience information",
        "**Acknowledgements**: Added new contributors and updated the contributor list"
    ]
    
    chapter_entries = [
        "**Chapter 3: DL Primer**: Added new diagrams explaining neural network architectures and improved explanations of backpropagation",
        "**Chapter 5: AI Workflow**: Enhanced the workflow diagram and added new examples for data preprocessing steps", 
        "**Chapter 8: AI Training**: Updated training examples with new code snippets and improved explanations of gradient descent",
        "**Chapter 1: Introduction**: Fixed several typos and improved the introduction to machine learning concepts",
        "**Chapter 12: Benchmarking AI**: Added new benchmarking metrics and updated performance comparison tables"
    ]
    
    lab_entries = [
        "**Lab: Arduino Image Classification**: Updated the image classification code with improved accuracy and added new examples",
        "**Lab: Raspberry Pi Setup**: Fixed setup instructions and added troubleshooting section for common issues"
    ]
    
    appendix_entries = [
        "**PhD Survival Guide**: Added new resources for graduate students and updated links"
    ]
    
    # Add impact bars
    frontmatter_with_impact = [f"- `███░░` {entry}" for entry in frontmatter_entries[:1]] + [f"- `██░░░` {entry}" for entry in frontmatter_entries[1:]]
    chapters_with_impact = [f"- `████░` {entry}" for entry in chapter_entries[:1]] + [f"- `███░░` {entry}" for entry in chapter_entries[1:3]] + [f"- `██░░░` {entry}" for entry in chapter_entries[3:]]
    labs_with_impact = [f"- `███░░` {entry}" for entry in lab_entries[:1]] + [f"- `██░░░` {entry}" for entry in lab_entries[1:]]
    appendix_with_impact = [f"- `█░░░░` {entry}" for entry in appendix_entries]
    
    demo_entry = f"""## {current_year} Updates

### 📅 {current_date}

<details>
<summary>**📄 Frontmatter**</summary>

{chr(10).join(frontmatter_with_impact)}

</details>

<details>
<summary>**📖 Chapters**</summary>

{chr(10).join(chapters_with_impact)}

</details>

<details>
<summary>**🧑‍💻 Labs**</summary>

{chr(10).join(labs_with_impact)}

</details>

<details>
<summary>**📚 Appendix**</summary>

{chr(10).join(appendix_with_impact)}

</details>
"""
    return demo_entry

def generate_release_notes_for_version(version, previous_version, description, verbose=False):
    """Generate release notes using your existing AI analysis"""
    
    print(f"📝 Generating release notes for {version}...")
    print(f"📋 Description: {description}")
    print(f"🔄 Previous version: {previous_version}")
    
    # Get the latest gh-pages commit date as the "since" date
    latest_commit, latest_date = get_latest_gh_pages_commit()
    
    if not latest_date:
        print("❌ No previous release found!")
        return None
    
    print(f"📅 Analyzing changes since: {format_friendly_date(latest_date)}")
    
    # Use your existing AI-powered analysis
    entry = generate_entry(latest_date, verbose=verbose, is_latest=True)
    
    if not entry:
        print("⚠️ No meaningful changes found")
        return None
    
    # Format as release notes instead of changelog
    release_notes = f"""## 📚 Release {version}

**{description}**

### 📋 Release Information
- **Type**: Release
- **Previous Version**: {previous_version}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Changes**: Since {format_friendly_date(latest_date)}

### 📝 What's New

{entry}

### 🔗 Quick Links
- 🌐 [Read Online](https://mlsysbook.ai)
- 📄 [Download PDF](https://mlsysbook.ai/pdf)
- 🧪 [Labs & Exercises](https://mlsysbook.ai/labs)
- 📚 [GitHub Repository](https://github.com/harvard-edge/cs249r_book)

### 📊 Technical Details
- **Build System**: Quarto with custom extensions
- **Deployment**: GitHub Pages + Netlify
- **PDF Generation**: LaTeX with compression
- **Content**: Markdown with interactive elements

---
*Generated with AI analysis of changes since last release*
"""
    
    print("✅ Release notes generated successfully")
    return release_notes

def fold_existing_entries(content):
    """Fold all existing details sections in the changelog content."""
    import re
    
    # Pattern to match <details open> and replace with <details>
    pattern = r'<details open>'
    replacement = '<details>'
    
    return re.sub(pattern, replacement, content)

def generate_changelog(mode="incremental", verbose=False):
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
        
        current_year = datetime.now().year
        year_header = f"## {current_year} Updates"
        return f"{year_header}\n\n{entry}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate changelog for ML systems book.")
    parser.add_argument("-f", "--full", action="store_true", help="Regenerate the entire changelog from scratch.")
    parser.add_argument("-u", "--update", action="store_true", help="Add new entries since last gh-pages publish.")
    parser.add_argument("-t", "--test", action="store_true", help="Run without writing to file.")
    parser.add_argument("--demo", action="store_true", help="Generate a demo changelog entry with sample data.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("-q", "--quarto-config", type=str, help="Path to quarto config file (default: book/config/_quarto-pdf.yml)")
    parser.add_argument("-m", "--model", type=str, default="gemma2:9b", help="Ollama model to use (default: gemma2:9b). Popular options: gemma2:9b, gemma2:27b, llama3.1:8b, llama3.1:70b")
    parser.add_argument("--release-notes", action="store_true", help="Generate release notes instead of changelog entry.")
    parser.add_argument("--version", type=str, help="Version for release notes (required with --release-notes).")
    parser.add_argument("--previous-version", type=str, help="Previous version for release notes (required with --release-notes).")
    parser.add_argument("--description", type=str, help="Release description (required with --release-notes).")

    args = parser.parse_args()
    
    # Handle demo mode first
    if args.demo:
        print("🎭 DEMO MODE - Generating sample changelog entry")
        demo_entry = generate_demo_entry()
        print("=" * 60)
        print("📝 DEMO CHANGELOG ENTRY")
        print("=" * 60)
        print(demo_entry)
        print("=" * 60)
        print("✅ Demo entry generated successfully!")
        exit(0)
    
    # Handle release notes mode
    if args.release_notes:
        if not args.version or not args.previous_version or not args.description:
            print("❌ Error: --release-notes requires --version, --previous-version, and --description")
            print("💡 Example: --release-notes --version v1.2.0 --previous-version v1.1.0 --description 'Add new chapter'")
            exit(1)
        
        print("📝 RELEASE NOTES MODE")
        mode = "release_notes"
    else:
        # Require either --full or --update to be specified
        if args.full and args.update:
            print("❌ Error: Cannot specify both --full and --update modes")
            exit(1)
        elif args.full:
            mode = "full"
        elif args.update:
            mode = "update"
        else:
            print("❌ Error: Must specify either --full, --update, or --release-notes mode")
            print("💡 Use --help for usage information")
            print("💡 Use --demo to see a sample changelog entry")
            exit(1)

    try:
        load_chapter_order(args.quarto_config)
        
        # Print configuration header
        print("=" * 60)
        print("📝 CHANGELOG GENERATION CONFIG")
        print("=" * 60)
        print(f"🎯 Mode: {mode.upper()}")
        print(f"🤖 AI Model: {args.model} (via Ollama)")
        print(f"🔧 Test Mode: {'ON' if args.test else 'OFF'}")
        print(f"📢 Verbose: {'ON' if args.verbose else 'OFF'}")
        print(f"📋 Features: Impact bars, importance sorting, specific summaries")
        print("=" * 60)
        print()
        
        print(f"🚀 Starting changelog generation in {mode} mode...")

        print(f"🤖 Using Ollama for summarization with model: {args.model}")
        use_ollama = True
        # Test Ollama connection
        test_response = call_ollama("Hello", model=args.model, verbose=False)
        if test_response is None:
            print("❌ Failed to connect to Ollama. Make sure it's running on localhost:11434")
            print("💡 To install models in Ollama:")
            print("   ollama pull gemma2:9b")
            print("   ollama pull gemma2:27b")
            exit(1)
        print("✅ Ollama connection successful")

        if mode == "release_notes":
            # Generate release notes
            new_entry = generate_release_notes_for_version(
                args.version, 
                args.previous_version, 
                args.description, 
                verbose=args.verbose
            )
        else:
            # Generate changelog entry
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

            if mode == "release_notes":
                # Save release notes to a file for the workflow to use
                release_notes_file = f"release_notes_{args.version}.md"
                with open(release_notes_file, "w", encoding="utf-8") as f:
                    f.write(new_entry.strip() + "\n")
                print(f"\n✅ Release notes written to {release_notes_file}")
                print("📋 Next step: Use this file in your GitHub workflow")
            else:
                # Save changelog entry
                with open(CHANGELOG_FILE, "w", encoding="utf-8") as f:
                    f.write(updated_content.strip() + "\n")
                print(f"\n✅ Changelog written to {CHANGELOG_FILE}")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()