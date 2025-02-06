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

def get_year_from_date(date_str):
    """Extract year from date string."""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z")
        return date_obj.year
    except ValueError:
        return None

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
        "# Book Changelog {.unnumbered}\n\n"
        "This Machine Learning Systems textbook originated from lecture materials developed for CS249r at Harvard. "
        "Through valuable feedback from students, professors, practitioners, researchers, and learners such as yourself, "
        "it continues to evolve and expand. While the content is useful in its current form, it remains in active development - "
        "with ongoing additions to technical coverage, enhanced explanations, and regular updates informed by academic and industry expertise. "
        "This changelog documents these improvements, helping you stay informed of the latest changes.\n\n"
        f"_Last Updated: {datetime.now().strftime('%b %d, %Y')}_\n\n"
    )

    # Ensure `gh-pages` branch exists on the remote
    gh_pages_exists = run_git_command(["git", "ls-remote", "--heads", "origin", "gh-pages"])
    if not gh_pages_exists:
        raise SystemExit("❌ Error: `gh-pages` branch not found on the remote. The changelog generation process requires this branch to exist.")

    # Fetch the `gh-pages` branch
    run_git_command(["git", "fetch", "origin", "gh-pages:refs/remotes/origin/gh-pages"])

    # Get commit history
    commits_with_dates = run_git_command([
        "git", "--no-pager", "log", "--pretty=format:%H %ad", "--date=iso", "origin/gh-pages"
    ]).split("\n")

    if not commits_with_dates:
        return intro_text + "_No `gh-pages` commits found._"

    # Parse commits and dates
    commits_with_dates = [(line.split(" ")[0], " ".join(line.split(" ")[1:])) for line in commits_with_dates]
    
    # Group changes by year
    changes_by_year = defaultdict(list)
    current_year = datetime.now().year
    first_details_created = False  # Track if we've created the first details section

    for i in range(len(commits_with_dates) - 1):
        current_commit, current_date = commits_with_dates[i]
        previous_commit, previous_date = commits_with_dates[i + 1]
        
        year = get_year_from_date(current_date)
        if not year:
            continue

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

        # Generate diff link
        full_diff_link = f"{GITHUB_REPO_URL}/compare/{previous_commit}...{current_commit}"
        
        # Create change entry
        change_entry = f"### 📅 Published on {format_friendly_date(current_date)}\n\n"
        change_entry += f"🔗 [View Full Diff]({full_diff_link})\n\n"

        total_added = sum(added for added, _ in changes_by_file.values())
        total_removed = sum(removed for _, removed in changes_by_file.values())
        total_files = len(changes_by_file)

        change_entry += f"- **{total_files} files updated**\n"
        change_entry += f"- **{total_added} lines added**, **{total_removed} lines removed**\n\n"

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

        if major_updates:
            details_open = " open" if not first_details_created else ""
            change_entry += f"<details{details_open}>\n"
            change_entry += "  <summary>**Major Updates**</summary>\n\n"
            change_entry += "\n".join(major_updates) + "\n\n"
            change_entry += "</details>\n\n"

        if minor_updates:
            details_open = " open" if not first_details_created else ""
            change_entry += f"<details{details_open}>\n"
            change_entry += "  <summary>**Minor Updates**</summary>\n\n"
            change_entry += "\n".join(minor_updates) + "\n\n"
            change_entry += "</details>\n\n"

        first_details_created = True
        changes_by_year[year].append(change_entry)

    # Build final summary
    summary = intro_text

    # Add each year's changes
    for year in sorted(changes_by_year.keys(), reverse=True):
        year_summary = f"## {year} Changes\n\n"
        year_summary += "\n".join(changes_by_year[year])        
        summary += year_summary
        summary += "\n---\n\n"

    return summary.strip()

if __name__ == "__main__":
    changelog = generate_changelog()
    with open(CHANGELOG_FILE, "w", encoding="utf-8") as f:
        f.write(changelog + "\n")
    print(f"📄 Changelog successfully generated in {CHANGELOG_FILE}")