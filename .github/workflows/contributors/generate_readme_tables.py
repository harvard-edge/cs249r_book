#!/usr/bin/env python3
"""
Generate All Contributors tables for README files.

This script reads .all-contributorsrc files and generates the HTML tables
that go in the README.md files.

Usage:
    python generate_readme_tables.py [--project PROJECT] [--update]
"""

import json
import re
import argparse
from pathlib import Path

PROJECTS = {
    "book": "book/",
    "kits": "kits/",
    "labs": "labs/",
    "tinytorch": "tinytorch/",
}

# Emoji mapping for contribution types (only types actually in use)
# Synced with generate_main_readme.py
EMOJI_KEY = {
    "bug": "🪲",             # Bug Hunter
    "code": "🧑‍💻",            # Code Contributor
    "design": "🎨",          # Design Artist
    "doc": "✍️",             # Doc Wizard
    "ideas": "🧠",           # Idea Spark
    "review": "🔎",          # Code Reviewer
    "test": "🧪",            # Test Tinkerer
    "tool": "🛠️",           # Tool Builder
}


def generate_contributor_cell(contributor: dict, image_size: int = 80) -> str:
    """Generate HTML for a single contributor cell."""
    login = contributor['login']
    name = contributor.get('name', login)
    avatar_url = contributor.get('avatar_url', f"https://avatars.githubusercontent.com/{login}")
    profile = contributor.get('profile', f"https://github.com/{login}")
    contributions = contributor.get('contributions', [])
    
    # Generate emoji badges
    badges = " ".join(EMOJI_KEY.get(c, c) for c in contributions)
    
    return f'''<td align="center" valign="top" width="14.28%"><a href="{profile}"><img src="{avatar_url}?v=4?s={image_size}" width="{image_size}px;" alt="{name}"/><br /><sub><b>{name}</b></sub></a><br />{badges}</td>'''


def generate_table(contributors: list[dict], per_line: int = 7, image_size: int = 80) -> str:
    """Generate the full HTML table for contributors."""
    if not contributors:
        return ""
    
    lines = ["<table>", "  <tbody>"]
    
    for i in range(0, len(contributors), per_line):
        row = contributors[i:i + per_line]
        lines.append("    <tr>")
        for contributor in row:
            lines.append("      " + generate_contributor_cell(contributor, image_size))
        lines.append("    </tr>")
    
    lines.append("  </tbody>")
    lines.append("</table>")
    
    return "\n".join(lines)


def update_readme(project_path: str, table_html: str) -> bool:
    """Update the README.md with the new contributors table."""
    readme_path = Path(project_path) / "README.md"
    
    if not readme_path.exists():
        print(f"Warning: {readme_path} does not exist")
        return False
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Pattern to match the ALL-CONTRIBUTORS-LIST section
    pattern = r'(<!-- ALL-CONTRIBUTORS-LIST:START.*?-->).*?(<!-- ALL-CONTRIBUTORS-LIST:END -->)'
    
    if not re.search(pattern, content, re.DOTALL):
        print(f"Warning: No ALL-CONTRIBUTORS-LIST markers found in {readme_path}")
        return False
    
    # Build replacement content
    replacement = f'''\\1
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
{table_html}

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

\\2'''
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open(readme_path, 'w') as f:
        f.write(new_content)
    
    return True


def process_project(project_name: str, project_path: str, update: bool = False) -> None:
    """Process a single project."""
    rc_path = Path(project_path) / ".all-contributorsrc"
    
    if not rc_path.exists():
        print(f"Skipping {project_name}: no .all-contributorsrc found")
        return
    
    with open(rc_path, 'r') as f:
        rc_data = json.load(f)
    
    contributors = rc_data.get('contributors', [])
    per_line = rc_data.get('contributorsPerLine', 7)
    image_size = rc_data.get('imageSize', 80)

    if not contributors:
        print(f"{project_name}: No contributors to display")
        return

    # Sort contributors by number of contributions (descending)
    sorted_contributors = sorted(
        contributors,
        key=lambda c: len(c.get('contributions', [])),
        reverse=True
    )

    table_html = generate_table(sorted_contributors, per_line, image_size)
    
    print(f"\n=== {project_name} ({len(contributors)} contributors) ===")
    
    if update:
        if update_readme(project_path, table_html):
            print(f"Updated {project_path}README.md")
        else:
            print(f"Failed to update {project_path}README.md")
    else:
        print(table_html)


def main():
    parser = argparse.ArgumentParser(description="Generate All Contributors tables")
    parser.add_argument("--project", choices=list(PROJECTS.keys()), help="Process specific project")
    parser.add_argument("--update", action="store_true", help="Update README files")
    args = parser.parse_args()

    # Find repo root (this script is in .github/workflows/contributors/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent.parent

    if args.project:
        projects = {args.project: PROJECTS[args.project]}
    else:
        projects = PROJECTS

    for name, rel_path in projects.items():
        process_project(name, str(repo_root / rel_path), args.update)


if __name__ == "__main__":
    main()
