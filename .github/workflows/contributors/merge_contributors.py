#!/usr/bin/env python3
"""
Merge contributor data from GitHub API with .all-contributorsrc names.

Produces a contributors.json file that the About page consumes.

Usage:
  python3 .github/workflows/contributors/merge_contributors.py

Typically run by GitHub Actions (weekly cron) or manually.
Requires: gh CLI authenticated, or GITHUB_TOKEN env var.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = "harvard-edge/cs249r_book"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT = REPO_ROOT / "site" / "about" / "contributors.json"
ALLCONTRIB = REPO_ROOT / ".all-contributorsrc"


def fetch_api_contributors() -> list[dict]:
    """Fetch contributors from GitHub API via gh CLI."""
    try:
        result = subprocess.run(
            [
                "gh", "api",
                f"repos/{REPO}/contributors",
                "--paginate",
                "--jq",
                '[.[] | select(.login != "github-actions[bot]" and .login != "dependabot[bot]") | {login, avatar_url, contributions, html_url}]'
            ],
            capture_output=True, text=True, check=True
        )
        # gh --paginate outputs multiple JSON arrays, one per page
        raw = result.stdout.strip()
        if not raw:
            return []

        # Parse potentially multiple JSON arrays
        contributors = []
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("["):
                contributors.extend(json.loads(line))
        return contributors
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error fetching from GitHub API: {e}", file=sys.stderr)
        return []


def load_allcontributors() -> dict[str, str]:
    """Load .all-contributorsrc to get login → real name mapping."""
    if not ALLCONTRIB.exists():
        return {}

    try:
        data = json.loads(ALLCONTRIB.read_text())
        return {
            c["login"]: c.get("name", c["login"])
            for c in data.get("contributors", [])
            if "login" in c
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: could not parse .all-contributorsrc: {e}", file=sys.stderr)
        return {}


def fetch_repo_stats() -> dict:
    """Fetch repository statistics."""
    try:
        result = subprocess.run(
            [
                "gh", "api", f"repos/{REPO}",
                "--jq",
                '{stars: .stargazers_count, forks: .forks_count, issues: .open_issues_count, watchers: .subscribers_count}'
            ],
            capture_output=True, text=True, check=True
        )
        return json.loads(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


def merge() -> None:
    """Main merge: combine API data with names, write JSON."""
    print("Fetching contributors from GitHub API...")
    api_data = fetch_api_contributors()
    print(f"  Found {len(api_data)} contributors from API.")

    print("Loading .all-contributorsrc for real names...")
    name_map = load_allcontributors()
    print(f"  Found {len(name_map)} name mappings.")

    print("Fetching repository stats...")
    stats = fetch_repo_stats()
    print(f"  Stars: {stats.get('stars', 'N/A')}, Forks: {stats.get('forks', 'N/A')}")

    # Merge: enrich API data with real names
    contributors = []
    for c in api_data:
        login = c["login"]
        contributors.append({
            "login": login,
            "name": name_map.get(login, login),
            "avatar_url": c["avatar_url"],
            "contributions": c["contributions"],
            "html_url": c["html_url"],
        })

    # Sort by contribution count (descending)
    contributors.sort(key=lambda x: x["contributions"], reverse=True)

    output = {
        "generated": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "repo": REPO,
        "stats": stats,
        "total": len(contributors),
        "contributors": contributors,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {len(contributors)} contributors to {OUTPUT}")


if __name__ == "__main__":
    merge()
