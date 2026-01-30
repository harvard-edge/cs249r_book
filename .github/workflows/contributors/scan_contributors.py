#!/usr/bin/env python3
"""
Scan git history to identify contributors per project.

This script analyzes git commits to:
1. Find unique contributors per project folder
2. Categorize contribution types based on commit messages and files changed
3. Map git emails to GitHub usernames where possible
4. Filter out bots and AI tools
5. Output data for .all-contributorsrc files

Usage:
    python scan_contributors.py [--project PROJECT] [--output json|table|rc]
    
Examples:
    python scan_contributors.py                      # Scan all projects
    python scan_contributors.py --project tinytorch  # Scan only tinytorch
    python scan_contributors.py --output json        # Output as JSON
"""

import subprocess
import json
import re
import argparse
from collections import defaultdict
from pathlib import Path

# Project folders to scan
PROJECTS = {
    "book": "book/",
    "kits": "kits/",
    "labs": "labs/",
    "tinytorch": "tinytorch/",
}

# Patterns to exclude (bots, AI tools, etc.)
EXCLUDE_PATTERNS = [
    r"bot",
    r"github-actions",
    r"dependabot",
    r"claude",
    r"cursor",
    r"copilot",
    r"\[bot\]",
    r"noreply\.github\.com.*bot",
]

# Contribution type detection based on commit message keywords
CONTRIBUTION_PATTERNS = {
    "bug": [
        r"\bfix(es|ed|ing)?\b",
        r"\bbug\b",
        r"\bissue\b",
        r"\berror\b",
        r"\bpatch\b",
        r"\bresolve[sd]?\b",
    ],
    "doc": [
        r"\bdoc(s|umentation)?\b",
        r"\breadme\b",
        r"\bcomment\b",
        r"\btypo\b",
        r"\bspelling\b",
        r"\bgrammar\b",
        r"\bexplain\b",
        r"\bdescription\b",
    ],
    "test": [
        r"\btest(s|ing)?\b",
        r"\bspec\b",
        r"\bcoverage\b",
        r"\bvalidat(e|ion)\b",
    ],
    "code": [
        r"\bfeat(ure)?\b",
        r"\badd(s|ed|ing)?\b",
        r"\bimplement(s|ed|ing|ation)?\b",
        r"\bcreate[sd]?\b",
        r"\bbuild\b",
        r"\brefactor\b",
        r"\bupdate[sd]?\b",
        r"\benhance\b",
        r"\bimprove[sd]?\b",
    ],
    "review": [
        r"\breview(ed|ing)?\b",
        r"\bfeedback\b",
        r"\bsuggestion\b",
    ],
    "design": [
        r"\bdesign\b",
        r"\bdiagram\b",
        r"\bfigure\b",
        r"\bimage\b",
        r"\billustrat(e|ion)\b",
        r"\bvisual\b",
    ],
    "translation": [
        r"\btranslat(e|ion|ed)\b",
        r"\blocali[sz](e|ation)\b",
        r"\bi18n\b",
    ],
    "tool": [
        r"\btool(s|ing)?\b",
        r"\bscript\b",
        r"\bautomation\b",
        r"\bworkflow\b",
        r"\bci\b",
        r"\bcd\b",
    ],
    "ideas": [
        r"\bidea\b",
        r"\bpropos(e|al)\b",
        r"\bsuggest\b",
        r"\brfc\b",
    ],
}

# Known email to GitHub username mappings (extend as needed)
EMAIL_TO_GITHUB = {
    # Core team
    "vj@eecs.harvard.edu": "profvjreddi",
    "zeljko.hrcek@gmail.com": "hzeljko",
    "mjrovai@gmail.com": "Mjrovai",
    "jjj4se@virginia.edu": "jasonjabbour",
    "iuchendu@g.harvard.edu": "uchendui",
    
    # Contributors
    "kkleinbard@avaya.com": "kai4avaya",
    "kai4avaya@gmail.com": "kai4avaya",
    "khoshnevis.naeem@gmail.com": "Naeemkh",
    "matthew_stewart@g.harvard.edu": "mrdragonbear",
    "jeffreyma@g.harvard.edu": "18jeffreyma",
    "douwedb@gmail.com": "V0XNIHILI",
    "shanzeh.batool@gmail.com": "shanzehbatool",
    "jaredping@yahoo.com": "JaredP94",
    "sara.khosravi.ds@gmail.com": "Sara-Khosravi",
    "i.j.shapira@gmail.com": "ishapira1",
    "durand.didier@gmail.com": "didier-durand",
    "gabriel.amazonas.eng@gmail.com": "GabrielAmazonas",
    "cbanbury@g.harvard.edu": "colbybanbury",
    "zishenwan@g.harvard.edu": "zishenwan",
    "mark@markmaz.com": "mmaz",
    "lazio2013@gmail.com": "ma3mool",
    "divya.amirtharaj@gmail.com": "DivyaAmirtharaj",
    "91.srivatsan@gmail.com": "srivatsankrishnan",
    "alexbrodriguez@gmail.com": "alxrod",
    "jlin3@college.harvard.edu": "jaysonzlin",
    "jwnchung@umich.edu": "jaywonchung",
    "jettythek@gmail.com": "jettythek",
    "anfe4949anfe@gmail.com": "andreamurillomtz",
    "oishibanerjee@gmail.com": "oishib",
    "yushun_hsiao@g.harvard.edu": "leo47007",
    "michael.schnebly@gmail.com": "MichaelSchnebly",
    "duongvanthuong9a8@gmail.com": "VThuong99",
    "eliasab@college.harvard.edu": "eliasab16",
}


def is_excluded(name: str, email: str) -> bool:
    """Check if contributor should be excluded (bot, AI, etc.)."""
    combined = f"{name} {email}".lower()
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return True
    return False


def detect_contribution_types(commit_message: str, files_changed: list[str]) -> set[str]:
    """Detect contribution types from commit message and files changed."""
    types = set()
    message_lower = commit_message.lower()
    
    # Check commit message patterns
    for contrib_type, patterns in CONTRIBUTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                types.add(contrib_type)
                break
    
    # Check file extensions for additional hints
    for file in files_changed:
        file_lower = file.lower()
        if file_lower.endswith(('.md', '.rst', '.txt')):
            types.add("doc")
        elif file_lower.endswith(('test_', '_test.py', 'test.py', '.spec.')):
            types.add("test")
        elif 'test' in file_lower:
            types.add("test")
        elif file_lower.endswith(('.png', '.jpg', '.svg', '.gif')):
            types.add("design")
        elif file_lower.endswith(('.py', '.js', '.ts', '.c', '.cpp', '.h')):
            types.add("code")
    
    # Default to "code" if nothing detected
    if not types:
        types.add("code")
    
    return types


def get_github_username(name: str, email: str) -> str | None:
    """Try to get GitHub username from email or name."""
    email_lower = email.lower()
    
    # Check known mappings
    if email_lower in EMAIL_TO_GITHUB:
        return EMAIL_TO_GITHUB[email_lower]
    
    # Try to extract from noreply email
    # Format: 12345+username@users.noreply.github.com
    noreply_match = re.match(r'(\d+\+)?([^@]+)@users\.noreply\.github\.com', email_lower)
    if noreply_match:
        return noreply_match.group(2)
    
    return None


def get_commits_for_project(project_path: str) -> list[dict]:
    """Get all commits for a project folder."""
    # Format: hash|author_name|author_email|subject
    cmd = [
        "git", "log",
        "--format=%H|%an|%ae|%s",
        "--name-only",
        "--", project_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return []
    
    commits = []
    current_commit = None
    
    for line in result.stdout.split('\n'):
        if '|' in line and line.count('|') >= 3:
            # This is a commit line
            if current_commit:
                commits.append(current_commit)
            
            parts = line.split('|', 3)
            current_commit = {
                'hash': parts[0],
                'name': parts[1],
                'email': parts[2],
                'message': parts[3] if len(parts) > 3 else '',
                'files': []
            }
        elif line.strip() and current_commit:
            # This is a file line
            current_commit['files'].append(line.strip())
    
    if current_commit:
        commits.append(current_commit)
    
    return commits


def analyze_project(project_name: str, project_path: str) -> dict:
    """Analyze a project and return contributor data."""
    commits = get_commits_for_project(project_path)
    
    # Aggregate by contributor
    contributors = defaultdict(lambda: {
        'name': '',
        'email': '',
        'github': None,
        'commits': 0,
        'types': set(),
    })
    
    for commit in commits:
        name = commit['name']
        email = commit['email']
        
        # Skip excluded contributors
        if is_excluded(name, email):
            continue
        
        # Use email as key for deduplication
        key = email.lower()
        
        contributors[key]['name'] = name
        contributors[key]['email'] = email
        contributors[key]['github'] = get_github_username(name, email)
        contributors[key]['commits'] += 1
        
        # Detect contribution types
        types = detect_contribution_types(commit['message'], commit['files'])
        contributors[key]['types'].update(types)
    
    # Convert to list and sort by commits
    result = []
    for email, data in contributors.items():
        result.append({
            'name': data['name'],
            'email': data['email'],
            'github': data['github'],
            'commits': data['commits'],
            'types': sorted(list(data['types'])),
        })
    
    result.sort(key=lambda x: x['commits'], reverse=True)
    return {
        'project': project_name,
        'path': project_path,
        'contributors': result,
        'total_contributors': len(result),
    }


def format_as_table(data: dict) -> str:
    """Format project data as a markdown table."""
    lines = [
        f"\n## {data['project']} ({data['total_contributors']} contributors)\n",
        "| Name | GitHub | Commits | Types |",
        "|------|--------|---------|-------|",
    ]
    
    for c in data['contributors']:
        github = f"@{c['github']}" if c['github'] else "?"
        types = ", ".join(c['types'])
        lines.append(f"| {c['name']} | {github} | {c['commits']} | {types} |")
    
    return "\n".join(lines)


def format_as_allcontributorsrc(data: dict) -> dict:
    """Format project data as .all-contributorsrc contributor entries."""
    # Dedupe by GitHub username
    seen_github = {}
    
    for c in data['contributors']:
        if not c['github']:
            continue  # Skip if no GitHub username
        
        github_lower = c['github'].lower()
        
        if github_lower in seen_github:
            # Merge contribution types
            seen_github[github_lower]['contributions'].update(c['types'])
            # Keep higher commit count name
            if c['commits'] > seen_github[github_lower].get('_commits', 0):
                seen_github[github_lower]['name'] = c['name']
                seen_github[github_lower]['_commits'] = c['commits']
        else:
            seen_github[github_lower] = {
                "login": c['github'],
                "name": c['name'],
                "avatar_url": f"https://avatars.githubusercontent.com/{c['github']}",
                "profile": f"https://github.com/{c['github']}",
                "contributions": set(c['types']),
                "_commits": c['commits']
            }
    
    # Convert sets to sorted lists and remove internal fields
    contributors = []
    for entry in seen_github.values():
        contributors.append({
            "login": entry['login'],
            "name": entry['name'],
            "avatar_url": entry['avatar_url'],
            "profile": entry['profile'],
            "contributions": sorted(list(entry['contributions']))
        })
    
    # Sort by number of contribution types (most active first)
    contributors.sort(key=lambda x: len(x['contributions']), reverse=True)
    
    return {
        "project": data['project'],
        "contributors": contributors
    }


def update_allcontributorsrc(project_name: str, contributors: list[dict], dry_run: bool = True) -> bool:
    """Update the .all-contributorsrc file for a project."""
    rc_path = Path(PROJECTS[project_name]) / ".all-contributorsrc"
    
    if not rc_path.exists():
        print(f"Warning: {rc_path} does not exist", file=__import__('sys').stderr)
        return False
    
    with open(rc_path, 'r') as f:
        rc_data = json.load(f)
    
    # Merge new contributors with existing
    existing_logins = {c['login'].lower() for c in rc_data.get('contributors', [])}
    
    added = []
    for new_contrib in contributors:
        if new_contrib['login'].lower() not in existing_logins:
            rc_data.setdefault('contributors', []).append(new_contrib)
            added.append(new_contrib['login'])
            existing_logins.add(new_contrib['login'].lower())
    
    if dry_run:
        print(f"\n[DRY RUN] Would add {len(added)} contributors to {rc_path}:")
        for login in added:
            print(f"  - @{login}")
        return True
    
    # Write updated file
    with open(rc_path, 'w') as f:
        json.dump(rc_data, f, indent=4)
    
    print(f"Updated {rc_path} with {len(added)} new contributors")
    return True


def main():
    parser = argparse.ArgumentParser(description="Scan git history for contributors")
    parser.add_argument("--project", choices=list(PROJECTS.keys()), help="Scan specific project")
    parser.add_argument("--output", choices=["json", "table", "rc"], default="table", help="Output format")
    parser.add_argument("--min-commits", type=int, default=1, help="Minimum commits to include")
    parser.add_argument("--update", action="store_true", help="Update .all-contributorsrc files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without writing")
    args = parser.parse_args()
    
    # Determine which projects to scan
    if args.project:
        projects = {args.project: PROJECTS[args.project]}
    else:
        projects = PROJECTS
    
    results = []
    
    for name, path in projects.items():
        print(f"Scanning {name}...", file=__import__('sys').stderr)
        data = analyze_project(name, path)
        
        # Filter by minimum commits
        data['contributors'] = [c for c in data['contributors'] if c['commits'] >= args.min_commits]
        data['total_contributors'] = len(data['contributors'])
        
        results.append(data)
    
    # Update mode
    if args.update or args.dry_run:
        for data in results:
            rc_data = format_as_allcontributorsrc(data)
            update_allcontributorsrc(
                data['project'], 
                rc_data['contributors'],
                dry_run=args.dry_run or not args.update
            )
        return
    
    # Output results
    if args.output == "json":
        print(json.dumps(results, indent=2))
    elif args.output == "rc":
        for data in results:
            rc_data = format_as_allcontributorsrc(data)
            print(f"\n=== {data['project']}/.all-contributorsrc contributors ===")
            print(json.dumps(rc_data['contributors'], indent=2))
    else:  # table
        for data in results:
            print(format_as_table(data))


if __name__ == "__main__":
    main()
