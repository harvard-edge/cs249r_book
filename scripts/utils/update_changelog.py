import subprocess
import re
import os
from collections import defaultdict
from datetime import datetime

CHANGELOG_FILE = "contents/frontmatter/changelog/changelog.qmd"
GITHUB_REPO_URL = "https://github.com/harvard-edge/cs249r_book/"
MAJOR_CHANGE_THRESHOLD = 200  # Define threshold for major updates

def format_friendly_date(date_str):
    """Format the date in a human-friendly way."""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z")
        return date_obj.strftime("%b %d, %Y")
    except ValueError:
        return date_str

def run_git_command(cmd):
    """Run a git command and return the output."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error running command: {' '.join(cmd)}")
        print(result.stderr)
        raise SystemExit(f"Git command failed: {' '.join(cmd)}")
    return result.stdout.strip()

def extract_chapter_title(file_path):
    """Extract the chapter title from the QMD file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    match = re.match(r"^#\s+(.*?)\s*(?:{.*)?$", line.strip())
                    if match:
                        return match.group(1).strip()
    except FileNotFoundError:
        pass
    return os.path.basename(file_path).replace("_", " ").replace(".qmd", "").title()

def get_changes_in_dev_since(date_start, date_end=None):
    """Get changes in the dev branch since a given date."""
    cmd = ["git", "log", "--numstat", "--since", date_start]
    if date_end:
        cmd += ["--until", date_end]
    cmd += ["dev", "--", "contents/**/*.qmd"]
    return run_git_command(cmd)

def generate_change_visual(added, removed, max_length=6):
    """Generate a visual representation of changes."""
    total = added + removed
    if total == 0:
        return ""
    added_blocks = int((added / total) * max_length) if total > 0 else 0
    removed_blocks = int((removed / total) * max_length) if total > 0 else 0
    added_bars = f'<span style="color:green">{"+" * added_blocks}</span>'
    removed_bars = f'<span style="color:red">{"-" * removed_blocks}</span>'
    return f"{added_bars}{removed_bars}"

def generate_changelog():
    """Generate the changelog content."""
    intro_text = (
        "---\n"
        "toc: false\n"
        "---\n\n"        
        "## Book Changelog {.unnumbered}\n\n"
        "This *Machine Learning Systems* textbook is constantly evolving. "
        "This changelog automatically records all updates and improvements, helping you stay informed about what's new and refined.\n\n"
    )

    # Ensure `gh-pages` branch exists, otherwise fail
    try:
        run_git_command(["git", "rev-parse", "--verify", "gh-pages"])
    except SystemExit:
        raise SystemExit("❌ Error: `gh-pages` branch not found. The changelog generation process requires this branch to exist.")

    # Get commit history from `gh-pages`
    commits_with_dates = run_git_command(["git", "--no-pager", "log", "--pretty=format:%H %ad", "--date=iso", "gh-pages"]).split("\n")
    
    if not commits_with_dates:
        return intro_text + "_No `gh-pages` commits found._"

    commits_with_dates = [(line.split(" ")[0], " ".join(line.split(" ")[1:])) for line in commits_with_dates]
    summary = intro_text  # Start the summary with the intro text

    for i in range(len(commits_with_dates) - 1):
        current_commit, current_date = commits_with_dates[i]
        previous_commit, previous_date = commits_with_dates[i + 1]

        changes = get_changes_in_dev_since(previous_date, current_date)
        if not changes.strip():
            continue

        changes_by_file = defaultdict(lambda: [0, 0])
        for line in changes.split("\n"):
            parts = line.split("\t")
            if len(parts) != 3:
                continue

            added, removed, file_path = parts
            added = int(added) if added.isdigit() else 0
            removed = int(removed) if removed.isdigit() else 0
            changes_by_file[file_path][0] += added
            changes_by_file[file_path][1] += removed

        # Generate diff link for the commit range
        full_diff_link = f"{GITHUB_REPO_URL}/compare/{previous_commit}...{current_commit}"
        summary += "---\n\n"
        summary += f"### 📅 Published on {format_friendly_date(current_date)}\n\n"
        summary += f"🔗 [View Full Diff]({full_diff_link})\n\n"

        total_added = sum(added for added, _ in changes_by_file.values())
        total_removed = sum(removed for _, removed in changes_by_file.values())
        total_files = len(changes_by_file)

        summary += f"- **{total_files} files updated**\n"
        summary += f"- **{total_added} lines added**, **{total_removed} lines removed**\n\n"

        # Separate Major and Minor Updates
        major_updates = []
        minor_updates = []

        for file_path, (added, removed) in changes_by_file.items():
            chapter_title = extract_chapter_title(file_path)
            total_changes = added + removed
            change_visual = generate_change_visual(added, removed)
            if total_changes > MAJOR_CHANGE_THRESHOLD:
                major_updates.append(f"- **{chapter_title}**: {change_visual} ({added} lines added, {removed} lines removed)")
            else:
                minor_updates.append(f"- **{chapter_title}**: {change_visual} ({added} lines added, {removed} lines removed)")

        # Add Major Updates Section
        if major_updates:
            summary += "<details>\n"
            summary += "  <summary>**Major Updates**</summary>\n\n"
            summary += "\n".join(major_updates) + "\n\n"
            summary += "</details>\n\n"

        # Add Minor Updates Section
        if minor_updates:
            summary += "<details>\n"
            summary += "  <summary>**Minor Updates**</summary>\n\n"
            summary += "\n".join(minor_updates) + "\n\n"
            summary += "</details>\n\n"

    summary += "\n---\n"
    return summary.strip()

if __name__ == "__main__":
    changelog = generate_changelog()
    with open(CHANGELOG_FILE, "w", encoding="utf-8") as f:
        f.write(changelog + "\n")
    print("📄 Changelog successfully generated in CHANGELOG.md!")
