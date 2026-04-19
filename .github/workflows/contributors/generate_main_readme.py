#!/usr/bin/env python3
"""
Generate the main README.md contributor section from all project configs.

This script reads each project's .all-contributorsrc file and generates a
sectioned contributor table for the main README.md.

To add or remove a project section, edit the PROJECT_SECTIONS list below —
that is the single source of truth used to produce the rendered output.

Usage:
    python generate_main_readme.py [--dry-run]
"""

import json
import re
import sys
from pathlib import Path

# Emoji mapping for contribution types (only types actually in use)
# Synced with generate_readme_tables.py
CONTRIBUTION_EMOJIS = {
    "bug": "🪲",             # Bug Hunter
    "code": "🧑‍💻",            # Code Contributor
    "design": "🎨",          # Design Artist
    "doc": "✍️",             # Doc Wizard
    "ideas": "🧠",           # Idea Spark
    "review": "🔎",          # Code Reviewer
    "test": "🧪",            # Test Tinkerer
    "tool": "🛠️",           # Tool Builder
}

# Legend for contribution types (shown in README)
# Only includes types currently in use across all projects
CONTRIBUTION_LEGEND = {
    "bug": ("🪲", "Bug Hunter"),
    "code": ("🧑‍💻", "Code Contributor"),
    "doc": ("✍️", "Doc Wizard"),
    "design": ("🎨", "Design Artist"),
    "ideas": ("🧠", "Idea Spark"),
    "review": ("🔎", "Code Reviewer"),
    "test": ("🧪", "Test Tinkerer"),
    "tool": ("🛠️", "Tool Builder"),
}


def load_config(path: Path) -> dict:
    """Load a .all-contributorsrc file."""
    if not path.exists():
        return {"contributors": []}
    with open(path) as f:
        return json.load(f)


def generate_contributor_cell(contributor: dict, show_badges: bool = True, image_size: int = 50, width_pct: str = "11.11%") -> str:
    """Generate HTML for a single contributor cell."""
    login = contributor.get("login", "")
    name = contributor.get("name", login)
    avatar_url = contributor.get("avatar_url", "")
    profile = contributor.get("profile", f"https://github.com/{login}")
    contributions = contributor.get("contributions", [])

    # Generate badge string
    badges = ""
    if show_badges and contributions:
        badges = " ".join(CONTRIBUTION_EMOJIS.get(c, "") for c in contributions)
        badges = f"<br />{badges}" if badges.strip() else ""

    return f'''      <td align="center" valign="top" width="{width_pct}"><a href="{profile}"><img src="{avatar_url}?v=4?s={image_size}" width="{image_size}px;" alt="{name}"/><br /><sub><b>{name}</b></sub></a>{badges}</td>'''


def generate_contributor_table(contributors: list, show_badges: bool = True, cols: int = 7, image_size: int = 50) -> str:
    """Generate an HTML table for contributors.

    Args:
        contributors: List of contributor dicts
        show_badges: Whether to show contribution badges
        cols: Number of columns per row (default 7 for readable wrapping)
        image_size: Size of avatar images in pixels (default 50 for compact display)
    """
    if not contributors:
        return "<p><em>Coming soon!</em></p>"

    # Sort by contribution count (most contributions first)
    sorted_contributors = sorted(
        contributors,
        key=lambda c: len(c.get("contributions", [])),
        reverse=True
    )

    # Calculate width percentage based on columns
    width_pct = f"{100/cols:.2f}%"

    rows = []
    row_cells = []

    for i, contributor in enumerate(sorted_contributors):
        row_cells.append(generate_contributor_cell(contributor, show_badges, image_size, width_pct))

        # Dynamic columns per row
        if len(row_cells) == cols:
            rows.append("    <tr>\n" + "\n".join(row_cells) + "\n    </tr>")
            row_cells = []

    # Add remaining cells
    if row_cells:
        rows.append("    <tr>\n" + "\n".join(row_cells) + "\n    </tr>")

    return f'''<table>
  <tbody>
{chr(10).join(rows)}
  </tbody>
</table>'''


def generate_legend() -> str:
    """Generate a compact legend for contribution types."""
    items = [f"{emoji} {title}" for emoji, title in CONTRIBUTION_LEGEND.values()]
    return " · ".join(items)


# Ordered list of project sections rendered in the main README.
# Each entry: (config_dir, emoji, title, marker_id)
#   - config_dir : top-level repo folder containing .all-contributorsrc
#                  (this is the on-disk directory, not necessarily the project
#                   key — e.g. "interviews" on disk renders as "StaffML")
#   - emoji      : section emoji
#   - title      : section heading text (after the emoji)
#   - marker_id  : prefix used in the HTML <!-- ${ID}-CONTRIBUTORS-START --> markers
PROJECT_SECTIONS = [
    ("book",        "📖",  "Textbook Contributors",        "BOOK"),
    ("tinytorch",   "🔥",  "TinyTorch Contributors",       "TINYTORCH"),
    ("mlsysim",     "🚀",  "MLSys·im Contributors",        "MLSYSIM"),
    ("interviews",  "🤖",  "StaffML Contributors",         "STAFFML"),
    ("kits",        "🛠️", "Hardware Kits Contributors",   "KITS"),
    ("labs",        "🧪",  "Labs Contributors",            "LABS"),
    ("slides",      "🎞️", "Slides Contributors",          "SLIDES"),
    ("instructors", "🗺️", "Instructor Site Contributors", "INSTRUCTORS"),
]


def _render_section(emoji: str, title: str, marker_id: str, table_html: str) -> str:
    return (
        f"### {emoji} {title}\n"
        "\n"
        f"<!-- {marker_id}-CONTRIBUTORS-START -->\n"
        "<!-- prettier-ignore-start -->\n"
        "<!-- markdownlint-disable -->\n"
        f"{table_html}\n"
        "\n"
        "<!-- markdownlint-restore -->\n"
        "<!-- prettier-ignore-end -->\n"
        f"<!-- {marker_id}-CONTRIBUTORS-END -->"
    )


def generate_sectioned_contributors(repo_root: Path) -> str:
    """Generate the full sectioned contributor section showing ALL contributors."""
    rendered_sections = []
    for config_dir, emoji, title, marker_id in PROJECT_SECTIONS:
        config = load_config(repo_root / config_dir / ".all-contributorsrc")
        contributors = config.get("contributors", [])
        table_html = generate_contributor_table(contributors)
        rendered_sections.append(_render_section(emoji, title, marker_id, table_html))

    legend = generate_legend()
    body = "\n\n---\n\n".join(rendered_sections)

    return (
        "## Contributors\n"
        "\n"
        "Thanks goes to these wonderful people who have contributed to making this resource better for everyone!\n"
        "\n"
        f"**Legend:** {legend}\n"
        "\n"
        f"{body}"
    )


def update_readme(repo_root: Path, dry_run: bool = False) -> bool:
    """Update the main README.md with sectioned contributors."""
    readme_path = repo_root / "README.md"

    if not readme_path.exists():
        print(f"ERROR: README.md not found at {readme_path}")
        return False

    content = readme_path.read_text()

    # Generate new contributor section
    new_section = generate_sectioned_contributors(repo_root)

    # Pattern to match the entire Contributors section
    # From "## Contributors" to just before the next "---" followed by a div or end of file
    pattern = r'## Contributors\n.*?(?=\n---\n\n<div align="center">|\Z)'

    if not re.search(pattern, content, re.DOTALL):
        print("ERROR: Could not find Contributors section in README.md")
        return False

    # Replace the section
    new_content = re.sub(pattern, new_section, content, flags=re.DOTALL)

    if dry_run:
        print("=== DRY RUN - Would update README.md with: ===")
        print(new_section[:2000] + "..." if len(new_section) > 2000 else new_section)
        return True

    readme_path.write_text(new_content)
    print(f"Updated {readme_path}")
    return True


def main():
    dry_run = "--dry-run" in sys.argv

    # Find repo root (this script is in .github/workflows/contributors/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent.parent

    # Verify we're in the right place
    if not (repo_root / "README.md").exists():
        print(f"ERROR: Cannot find README.md in {repo_root}")
        sys.exit(1)

    success = update_readme(repo_root, dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
