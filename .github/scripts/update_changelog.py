import subprocess
import re
import os
import argparse
import yaml
from collections import defaultdict
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHANGELOG_FILE = "CHANGELOG.md"
QUARTO_YML_FILE = "_quarto.yml"
GITHUB_REPO_URL = "https://github.com/harvard-edge/cs249r_book/"
MAJOR_CHANGE_THRESHOLD = 200

chapter_order = []

chapter_lookup = [
    ("introduction.qmd", "Introduction", 1),
    ("ml_systems.qmd", "ML Systems", 2),
    ("dl_primer.qmd", "DL Primer", 3),
    ("dnn_architectures.qmd", "DNN Architectures", 4),
    ("workflow.qmd", "AI Workflow", 5),
    ("data_engineering.qmd", "Data Engineering", 6),
    ("frameworks.qmd", "AI Frameworks", 7),
    ("training.qmd", "AI Training", 8),
    ("efficient_ai.qmd", "Efficient AI", 9),
    ("optimizations.qmd", "Model Optimizations", 10),
    ("hw_acceleration.qmd", "AI Acceleration", 11),
    ("benchmarking.qmd", "Benchmarking AI", 12),
    ("ops.qmd", "ML Operations", 13),
    ("ondevice_learning.qmd", "On-Device Learning", 14),
    ("privacy_security.qmd", "Security & Privacy", 15),
    ("responsible_ai.qmd", "Responsible AI", 16),
    ("sustainable_ai.qmd", "Sustainable AI", 17),
    ("robust_ai.qmd", "Robust AI", 18),
    ("ai_for_good.qmd", "AI for Good", 19),
    ("conclusion.qmd", "Conclusion", 20),
]

def load_chapter_order():
    global chapter_order
    with open(QUARTO_YML_FILE, "r", encoding="utf-8") as f:
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

    print(f"📚 Loaded {len(chapter_order)} chapters from _quarto.yml")

def run_git_command(cmd, verbose=False):
    if verbose:
        print(f"📦 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Git command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()

def extract_chapter_title(file_path):
    base = os.path.basename(file_path)
    for fname, title, number in chapter_lookup:
        if fname == base:
            return f"Chapter {number}: {title}"
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

def summarize_changes_with_openai(file_path, commit_messages, verbose=False):
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

    try:
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

        return f"- **{chapter_title}**: {clean_summary}"

    except Exception as e:
        print(f"⚠️ OpenAI failed for {file_path}: {e}")
        return f"- **{chapter_title}**: _(unable to summarize; see commits manually)_"


def format_friendly_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z").strftime("%b %d, %Y")
    except:
        return date_str

def normalized_path(path):
    return os.path.normpath(path).lower()

def generate_entry(start_date, end_date=None, verbose=False):
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
    entry = f"### 📅 Published on {current_date}\n\n"

    major, minor = [], []

    ordered_files = sorted(
        changes_by_file,
        key=lambda f: next(
            (i for i, ch in enumerate(chapter_order) if normalized_path(f).endswith(normalized_path(ch))),
            float('inf')
        )
    )

    for file_path in ordered_files:
        total = added + removed
        if verbose:
            print(f"🔍 Summarizing {file_path} ({added}+ / {removed}-)")
        commit_msgs = get_commit_messages_for_file(file_path, start_date, end_date, verbose=verbose)
        summary = summarize_changes_with_openai(file_path, commit_msgs, verbose=verbose)
        if total > MAJOR_CHANGE_THRESHOLD:
            major.append(summary)
        else:
            minor.append(summary)

    if major:
        entry += "<details open>\n<summary>**Major Updates**</summary>\n\n" + "\n".join(sort_by_chapter_order(major)) + "\n\n</details>\n\n"
    if minor:
        entry += "<details open>\n<summary>**Minor Updates**</summary>\n\n" + "\n".join(sort_by_chapter_order(minor)) + "\n\n</details>\n"

    return entry

def generate_changelog(mode="incremental", verbose=False):
    run_git_command(["git", "fetch", "origin", "gh-pages:refs/remotes/origin/gh-pages"], verbose=verbose)
    run_git_command(["git", "fetch", "origin", "dev:refs/remotes/origin/dev"], verbose=verbose)

    def get_latest_gh_pages_commit():
        output = run_git_command(["git", "log", "-1", "--pretty=format:%H %ad", "--date=iso", "origin/gh-pages"], verbose=verbose)
        parts = output.split(" ", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else (None, None)

    latest_commit, latest_date = get_latest_gh_pages_commit()
    intro = f""
    sections = []

    if mode == "full":
        if verbose:
            print("🔁 Running full regeneration...")
        commits = run_git_command(["git", "log", "--pretty=format:%H %ad", "--date=iso", "origin/gh-pages"], verbose=verbose).splitlines()
        commits = [(c.split(" ")[0], " ".join(c.split(" ")[1:])) for c in commits]
        for i in range(len(commits) - 1):
            entry = generate_entry(commits[i + 1][1], commits[i][1], verbose=verbose)
            if entry:
                sections.append(entry)
    else:
        if verbose:
            print("⚡ Running incremental update...")
        entry = generate_entry(latest_date, verbose=verbose)
        if entry:
            sections.append(entry)

    if not sections:
        return intro + "_No updates found._"

    year = datetime.now().year
    return "\n\n".join(sections) + "\n"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate changelog for ML systems book.")
    parser.add_argument("-i", "--incremental", action="store_true", help="Add new entries since last gh-pages publish (default).")
    parser.add_argument("-f", "--full", action="store_true", help="Regenerate the entire changelog from scratch.")
    parser.add_argument("-t", "--test", action="store_true", help="Run without writing to file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")

    args = parser.parse_args()
    mode = "incremental"
    if args.full:
        mode = "full"

    try:
        load_chapter_order()
        new_entry = generate_changelog(mode=mode, verbose=args.verbose)  # returns just the `📅 Published on ...` block

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
            updated_content = f"{year_header}\n\n{new_entry.strip()}\n---\n\n{cleaned_existing}"

            with open(CHANGELOG_FILE, "w", encoding="utf-8") as f:
                f.write(updated_content.strip() + "\n")

            print(f"\n✅ Changelog written to {CHANGELOG_FILE}")
    except Exception as e:
        print(f"❌ Error: {e}")

