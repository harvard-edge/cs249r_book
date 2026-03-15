#!/usr/bin/env python3
"""
Generate the 'Trending Community Questions' list for the Interview Hub.

This script queries the GitHub API for issues labeled 'interview-prep',
sorts them by upvote (+1) reactions, and updates interviews/README.md.

Usage:
    python generate_interview_leaderboard.py
"""

import os
import re
import sys
import requests
from pathlib import Path

OWNER = "harvard-edge"
REPO = "cs249r_book"
LABEL = "interview-prep"

def fetch_trending_issues(token):
    headers = {"Authorization": f"token {token}"}
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues?labels={LABEL}&state=open"
    
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        print(f"Error fetching issues: {res.status_code}")
        return []
    
    issues = res.json()
    trending = []
    
    for issue in issues:
        # Ignore pull requests (which are also returned by the issues API)
        if "pull_request" in issue:
            continue
            
        upvotes = issue.get("reactions", {}).get("+1", 0)
        trending.append({
            "title": issue["title"],
            "url": issue["html_url"],
            "upvotes": upvotes,
            "author": issue["user"]["login"]
        })
    
    # Sort by upvotes (descending)
    trending.sort(key=lambda x: x["upvotes"], reverse=True)
    return trending

def generate_markdown_table(issues):
    if not issues:
        return "*No trending questions yet. Be the first to [submit one](https://github.com/harvard-edge/cs249r_book/issues/new?template=interview_question.yml)!*"
    
    lines = ["| Question | Upvotes | Author |", "| :--- | :---: | :--- |"]
    for issue in issues[:10]:  # Show top 10
        title = issue["title"].replace("Interview Question: ", "")
        lines.append(f"| [{title}]({issue['url']}) | {issue['upvotes']} 👍 | @{issue['author']} |")
    
    return "
".join(lines)

def update_readme(repo_root, markdown):
    readme_path = repo_root / "interviews" / "README.md"
    if not readme_path.exists():
        print(f"README not found at {readme_path}")
        return
    
    content = readme_path.read_text()
    
    # Pattern to match the Trending Questions section
    pattern = r"(<!-- TRENDING-QUESTIONS-START -->).*(<!-- TRENDING-QUESTIONS-END -->)"
    replacement = f"\1
{markdown}
\2"
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    readme_path.write_text(new_content)
    print("Updated interviews/README.md with trending questions.")

def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not found. Skipping leaderboard generation.")
        return

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent.parent
    
    issues = fetch_trending_issues(token)
    markdown = generate_markdown_table(issues)
    update_readme(repo_root, markdown)

if __name__ == "__main__":
    main()
