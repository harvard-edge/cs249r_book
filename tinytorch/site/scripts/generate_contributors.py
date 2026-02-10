#!/usr/bin/env python3
"""
Generate contributor data for TinyTorch site.

Fetches contributor info from GitHub API and outputs markdown/JSON for the site.
Run this in CI or manually to update the contributors list.

Usage:
    python3 scripts/generate_contributors.py > _data/contributors.json
    python3 scripts/generate_contributors.py --markdown >> credits.md
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class Contributor:
    login: str
    contributions: int
    avatar_url: str
    html_url: str
    name: Optional[str] = None


def fetch_contributors(repo: str = "harvard-edge/cs249r_book", limit: int = 50) -> list[Contributor]:
    """Fetch contributors from GitHub API using gh CLI."""
    cmd = [
        "gh", "api", f"repos/{repo}/contributors",
        "--paginate",
        "--jq", '.[] | select(.type == "User") | {login, contributions, avatar_url, html_url}'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error fetching contributors: {result.stderr}", file=sys.stderr)
        return []

    contributors = []
    for line in result.stdout.strip().split('\n'):
        if line:
            data = json.loads(line)
            contributors.append(Contributor(**data))

    # Sort by contributions and limit
    contributors.sort(key=lambda c: c.contributions, reverse=True)
    return contributors[:limit]


def fetch_user_names(contributors: list[Contributor]) -> list[Contributor]:
    """Fetch real names for top contributors."""
    for c in contributors[:20]:  # Only fetch names for top 20 to avoid rate limits
        cmd = ["gh", "api", f"users/{c.login}", "--jq", ".name"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            c.name = result.stdout.strip()
    return contributors


def generate_json(contributors: list[Contributor]) -> str:
    """Generate JSON output."""
    return json.dumps([
        {
            "login": c.login,
            "name": c.name or c.login,
            "contributions": c.contributions,
            "avatar_url": c.avatar_url,
            "html_url": c.html_url
        }
        for c in contributors
    ], indent=2)


def generate_markdown(contributors: list[Contributor]) -> str:
    """Generate markdown contributor grid."""
    lines = []
    lines.append("<!-- AUTO-GENERATED: Do not edit manually. Run scripts/generate_contributors.py -->")
    lines.append("")
    lines.append("```{raw} html")
    lines.append('<div class="contributor-grid">')

    for c in contributors:
        display_name = c.name or c.login
        lines.append(f'''  <a href="{c.html_url}" class="contributor" title="{display_name} ({c.contributions} contributions)">
    <img src="{c.avatar_url}&s=80" alt="{c.login}" />
    <span class="name">{display_name}</span>
    <span class="commits">{c.contributions}</span>
  </a>''')

    lines.append('</div>')
    lines.append("```")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate contributor data")
    parser.add_argument("--markdown", action="store_true", help="Output markdown instead of JSON")
    parser.add_argument("--limit", type=int, default=50, help="Max contributors to include")
    parser.add_argument("--with-names", action="store_true", help="Fetch real names (slower)")
    args = parser.parse_args()

    contributors = fetch_contributors(limit=args.limit)

    if args.with_names:
        contributors = fetch_user_names(contributors)

    if args.markdown:
        print(generate_markdown(contributors))
    else:
        print(generate_json(contributors))


if __name__ == "__main__":
    main()
