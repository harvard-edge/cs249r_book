#!/usr/bin/env python3
"""
Simple Git Cleanup Tool - Show Recent and Large Files
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

def run_git_command(cmd, cwd=None):
    """Run a git command and return output"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or Path.cwd(),
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return ""

def get_recent_files(days=7):
    """Get files modified in the last N days"""
    print(f"🔍 Finding files modified in the last {days} days...")

    since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    cmd = [
        "git", "log", "--since", since_date,
        "--name-only", "--pretty=format:",
        "--diff-filter=M"
    ]

    output = run_git_command(cmd)
    if not output:
        return []

    # Parse the output to get unique files
    files = set()
    for line in output.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('commit') and not line.startswith('Author'):
            files.add(line)

    # Get additional info for each file
    file_info = []
    for file_path in sorted(files):
        if os.path.exists(file_path):
            try:
                stat = os.stat(file_path)
                file_info.append({
                    'path': file_path,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                })
            except Exception as e:
                print(f"Error getting info for {file_path}: {e}")

    return file_info

def get_large_files(min_size_mb=10):
    """Find large files in the repository"""
    print(f"🔍 Finding files larger than {min_size_mb}MB...")

    find_cmd = [
        "find", ".", "-type", "f", "-size", f"+{min_size_mb}M",
        "-not", "-path", "./.git/*"
    ]

    output = run_git_command(find_cmd)
    if not output:
        return []

    file_info = []
    for line in output.strip().split('\n'):
        if line.strip():
            file_path = line.strip()
            if os.path.exists(file_path):
                try:
                    stat = os.stat(file_path)
                    file_info.append({
                        'path': file_path,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                    })
                except Exception as e:
                    print(f"Error getting info for {file_path}: {e}")

    return file_info

def display_files_table(files, title):
    """Display files in a table"""
    if not files:
        print(f"⚠️  No files found for: {title}")
        return

    print(f"\n{title}")
    print("=" * 80)
    print(f"{'#':<4} {'File Path':<50} {'Size':<12} {'Modified':<20}")
    print("-" * 80)

    for i, file_info in enumerate(files, 1):
        size_mb = file_info['size'] / (1024 * 1024)
        size_str = f"{size_mb:.1f}MB" if size_mb >= 1 else f"{file_info['size'] / 1024:.1f}KB"

        path = file_info['path']
        if len(path) > 48:
            path = path[:45] + "..."

        print(f"{i:<4} {path:<50} {size_str:<12} {file_info['modified']:<20}")

def get_git_stats():
    """Get git repository statistics"""
    print("📊 Repository Statistics")
    print("=" * 40)

    # Get commit count
    cmd = ["git", "rev-list", "--count", "HEAD"]
    output = run_git_command(cmd)
    if output:
        print(f"Total commits: {output.strip()}")

    # Get repository size
    cmd = ["git", "count-objects", "-vH"]
    output = run_git_command(cmd)
    if output:
        for line in output.strip().split('\n'):
            if 'size-pack' in line:
                print(f"Pack size: {line.split(':')[1].strip()}")
            elif 'size-garbage' in line:
                print(f"Garbage size: {line.split(':')[1].strip()}")

def main():
    print("🗑️  Git Cleanup Tool - File Analysis")
    print("=" * 50)

    # Check if we're in a git repo
    if not os.path.exists('.git'):
        print("❌ Not a git repository. Please run this from a git repo.")
        return

    # Show repository stats
    get_git_stats()

    # Get recent files (last 7 days)
    recent_files = get_recent_files(days=7)
    display_files_table(recent_files, "Files Modified in Last 7 Days")

    # Get large files (>10MB)
    large_files = get_large_files(min_size_mb=10)
    display_files_table(large_files, "Files Larger Than 10MB")

    # Show summary
    print(f"\n📋 Summary:")
    print(f"  Recent files: {len(recent_files)}")
    print(f"  Large files: {len(large_files)}")

    if recent_files or large_files:
        print(f"\n💡 To remove files from git history, you can use:")
        print(f"   git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch <file>' --prune-empty --tag-name-filter cat -- --all")
        print(f"   git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin")
        print(f"   git reflog expire --expire=now --all")
        print(f"   git gc --prune=now --aggressive")
    else:
        print("✅ No files found that need cleanup!")

if __name__ == "__main__":
    main()
