import subprocess
import re
import os
import argparse
import yaml
import time
from collections import defaultdict
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHANGELOG_FILE = "CHANGELOG.md"
QUARTO_YML_FILE = "book/config/_quarto-pdf.yml"  # Default to PDF config which has chapters structure
GITHUB_REPO_URL = "https://github.com/harvard-edge/cs249r_book/"
# Removed MAJOR_CHANGE_THRESHOLD since we're organizing by content type now
OPENAI_DELAY = 1  # seconds between API calls

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
    ("contents/frontmatter/ai/socratiq.qmd", "SocraticAI", 204),
    
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

def sort_by_chapter_order(updates):
    def extract_path(update):
        match = re.search(r'\*\*(.*?)\*\*', update)
        if match:
            title = match.group(1)
            for path in chapter_order:
                if title.lower().replace(' ', '_') in path.lower():
                    return chapter_order.index(path)
        return float('inf')
    return sorted(updates, key=extract_path)

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
    return run_git_command(cmd, verbose=verbose)

def summarize_changes_with_openai(file_path, commit_messages, verbose=False, max_retries=3):
    chapter_title = extract_chapter_title(file_path)
    if verbose:
        print(f"🤖 Calling OpenAI for: {file_path} -- {chapter_title}")

    prompt = f"""You're helping to generate a changelog for a machine learning systems textbook.
The following file has been updated: {file_path}

Here are the commit messages:
{commit_messages}

Summarize the meaningful content-level changes (new sections, rewrites, example additions, figure changes).
Ignore formatting or typo-only changes.

Only return the summary sentence (not the bullet or chapter title)."""

    for attempt in range(max_retries):
        try:
            # Add delay to avoid rate limiting
            if attempt > 0:
                time.sleep(OPENAI_DELAY * (2 ** attempt))  # exponential backoff
            
            response = client.chat.completions.create(
                model="gpt-4",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant writing changelog summaries."},
                    {"role": "user", "content": prompt}
                ]
            )

            summary = response.choices[0].message.content.strip()

            if not summary:
                return f"- **{chapter_title}**: _(no meaningful changes detected)_"

            clean_summary = summary.partition(":")[-1].strip()
            if not clean_summary:
                clean_summary = summary  # fallback if no colon was present

            # Add delay after successful call
            time.sleep(OPENAI_DELAY)
            return f"- **{chapter_title}**: {clean_summary}"

        except Exception as e:
            print(f"⚠️ OpenAI attempt {attempt + 1} failed for {file_path}: {e}")
            if attempt == max_retries - 1:
                return f"- **{chapter_title}**: _(unable to summarize; see commits manually)_"

def format_friendly_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z").strftime("%b %d, %Y")
    except:
        return date_str

def normalized_path(path):
    return os.path.normpath(path).lower()

def generate_entry(start_date, end_date=None, verbose=False, is_latest=False):
    if verbose:
        print(f"📁 Processing changes from {start_date} to {end_date or 'now'}")
    changes = get_changes_in_dev_since(start_date, end_date, verbose=verbose)
    if not changes.strip():
        return None

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

    current_date = datetime.now().strftime('%b %d, %Y') if not end_date else format_friendly_date(end_date)
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
    for idx, file_path in enumerate(ordered_files, 1):
        added, removed = changes_by_file[file_path]
        total = added + removed
        if verbose:
            print(f"🔍 Summarizing {file_path} ({added}+ / {removed}-) [{idx}/{total_files}]")
        
        # Skip references
        if "references.qmd" in file_path:
            continue
            
        commit_msgs = get_commit_messages_for_file(file_path, start_date, end_date, verbose=verbose)
        summary = summarize_changes_with_openai(file_path, commit_msgs, verbose=verbose)
        
        # Categorize by content type
        if "contents/frontmatter/" in file_path:
            frontmatter.append(summary)
        elif "contents/labs/" in file_path:
            labs.append(summary)
        elif "contents/appendix/" in file_path:
            appendix.append(summary)
        else:
            chapters.append(summary)

    # Determine if sections should be open or closed
    # Only the latest entry should be open, all previous entries should be folded
    details_state = "open" if is_latest else ""

    # Add sections in order: Frontmatter, Chapters, Labs, Appendix
    if frontmatter:
        entry += f"<details {details_state}>\n<summary>**📄 Frontmatter**</summary>\n\n" + "\n".join(sort_by_chapter_order(frontmatter)) + "\n\n</details>\n\n"
    if chapters:
        entry += f"<details {details_state}>\n<summary>**📖 Chapters**</summary>\n\n" + "\n".join(sort_by_chapter_order(chapters)) + "\n\n</details>\n\n"
    if labs:
        entry += f"<details {details_state}>\n<summary>**🧑‍💻 Labs**</summary>\n\n" + "\n".join(sort_by_chapter_order(labs)) + "\n\n</details>\n\n"
    if appendix:
        entry += f"<details {details_state}>\n<summary>**📚 Appendix**</summary>\n\n" + "\n".join(sort_by_chapter_order(appendix)) + "\n\n</details>\n"

    return entry

def fold_existing_entries(content):
    """Fold all existing details sections in the changelog content."""
    import re
    
    # Pattern to match <details open> and replace with <details>
    pattern = r'<details open>'
    replacement = '<details>'
    
    return re.sub(pattern, replacement, content)

def generate_changelog(mode="incremental", verbose=False):
    print("🔄 Fetching latest Git data...")
    run_git_command(["git", "fetch", "origin", "gh-pages:refs/remotes/origin/gh-pages"], verbose=verbose)
    run_git_command(["git", "fetch", "origin", "dev:refs/remotes/origin/dev"], verbose=verbose)

    def get_latest_gh_pages_commit():
        # Look for actual publication commits, not administrative ones
        output = run_git_command(["git", "log", "--pretty=format:%H %ad", "--date=iso", "--grep=Built site for gh-pages", "origin/gh-pages"], verbose=verbose)
        if output.strip():
            first_line = output.split('\n')[0]
            parts = first_line.split(" ", 1)
            return (parts[0], parts[1]) if len(parts) == 2 else (None, None)
        return (None, None)

    def get_all_gh_pages_commits():
        # Look for actual publication commits, not administrative ones
        output = run_git_command(["git", "log", "--pretty=format:%H %ad", "--date=iso", "--grep=Built site for gh-pages", "origin/gh-pages"], verbose=verbose)
        commits = []
        for line in output.splitlines():
            parts = line.split(" ", 1)
            if len(parts) == 2:
                commits.append((parts[0], parts[1]))
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
            year_header = f"## {year} Changes"
            year_entries = "\n\n".join(entries_by_year[year])
            output_sections.append(f"{year_header}\n\n{year_entries}")
        
        return "\n\n---\n\n".join(output_sections) + "\n"
        
    else:
        if verbose:
            print("⚡ Running incremental update...")
        entry = generate_entry(latest_date, verbose=verbose, is_latest=True)
        if not entry:
            return "_No updates found._"
        
        current_year = datetime.now().year
        year_header = f"## {current_year} Changes"
        return f"{year_header}\n\n{entry}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate changelog for ML systems book.")
    parser.add_argument("-i", "--incremental", action="store_true", help="Add new entries since last gh-pages publish (default).")
    parser.add_argument("-f", "--full", action="store_true", help="Regenerate the entire changelog from scratch.")
    parser.add_argument("-t", "--test", action="store_true", help="Run without writing to file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("-q", "--quarto-config", type=str, help="Path to quarto config file (default: book/config/_quarto-pdf.yml)")

    args = parser.parse_args()
    mode = "incremental"
    if args.full:
        mode = "full"

    try:
        load_chapter_order(args.quarto_config)
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
            year_header = f"## {current_year} Changes"

            # Remove first occurrence of the year header
            existing_lines = existing.splitlines()
            filtered_lines = []
            found = False
            for line in existing_lines:
                if not found and line.strip() == year_header:
                    found = True
                    continue  # skip this one line only
                filtered_lines.append(line)

            cleaned_existing = "\n".join(filtered_lines).strip()

            # Prepend the new entry with correct year section
            if mode == "full":
                # For full mode, replace entire content (already includes year headers)
                updated_content = new_entry.strip()
            else:
                # For incremental, prepend to existing and fold all previous entries
                folded_existing = fold_existing_entries(cleaned_existing)
                updated_content = f"{new_entry.strip()}\n---\n\n{folded_existing}"

            with open(CHANGELOG_FILE, "w", encoding="utf-8") as f:
                f.write(updated_content.strip() + "\n")

            print(f"\n✅ Changelog written to {CHANGELOG_FILE}")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()