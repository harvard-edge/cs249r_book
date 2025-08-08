#!/usr/bin/env python3
"""
GitHub Workflow Runs Cleanup Script

This script helps clean up old GitHub workflow runs while keeping a configurable
number of recent runs per workflow. Useful for cleaning up failed debugging runs.

Usage:
    python cleanup_workflow_runs.py --help
    python cleanup_workflow_runs.py --dry-run
    python cleanup_workflow_runs.py --keep 5 --token YOUR_TOKEN
    python cleanup_workflow_runs.py --keep 10 --workflow "quarto-build.yml"

Requirements:
    - GitHub personal access token with 'actions:write' scope
    - Set token via --token flag or GITHUB_TOKEN environment variable
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests


class GitHubWorkflowCleaner:
    """Manages cleanup of GitHub workflow runs."""
    
    def __init__(self, token: str, repo: str, dry_run: bool = False):
        """
        Initialize the workflow cleaner.
        
        Args:
            token: GitHub personal access token
            repo: Repository in format 'owner/repo'
            dry_run: If True, only preview actions without executing
        """
        self.token = token
        self.repo = repo
        self.dry_run = dry_run
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MLSysBook-Workflow-Cleaner"
        }
        
    def _make_request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """Make a GitHub API request with error handling."""
        try:
            response = requests.request(method, url, headers=self.headers, **kwargs)
            if response.status_code == 403:
                # Check for rate limiting
                if 'X-RateLimit-Remaining' in response.headers:
                    remaining = int(response.headers['X-RateLimit-Remaining'])
                    if remaining == 0:
                        reset_time = int(response.headers['X-RateLimit-Reset'])
                        wait_time = reset_time - int(time.time()) + 1
                        print(f"⚠️  Rate limit exceeded. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        return self._make_request(method, url, **kwargs)
                print(f"❌ Permission denied. Check your token has 'actions:write' scope.")
                return None
            elif response.status_code == 404:
                print(f"❌ Repository not found: {self.repo}")
                return None
            elif not response.ok:
                print(f"❌ API request failed: {response.status_code} - {response.text}")
                return None
            return response
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def get_workflows(self) -> List[Dict]:
        """Get all workflows in the repository."""
        url = f"{self.base_url}/repos/{self.repo}/actions/workflows"
        response = self._make_request("GET", url)
        if not response:
            return []
        
        workflows = response.json().get('workflows', [])
        print(f"📋 Found {len(workflows)} workflows")
        return workflows
    
    def get_workflow_runs(self, workflow_id: str, per_page: int = 100) -> List[Dict]:
        """Get all runs for a specific workflow."""
        all_runs = []
        page = 1
        
        while True:
            url = f"{self.base_url}/repos/{self.repo}/actions/workflows/{workflow_id}/runs"
            params = {
                "per_page": per_page,
                "page": page
            }
            
            response = self._make_request("GET", url, params=params)
            if not response:
                break
                
            data = response.json()
            runs = data.get('workflow_runs', [])
            
            if not runs:
                break
                
            all_runs.extend(runs)
            
            # Check if we have more pages
            if len(runs) < per_page:
                break
                
            page += 1
            
        return all_runs
    
    def delete_workflow_run(self, run_id: str) -> bool:
        """Delete a specific workflow run."""
        if self.dry_run:
            return True
            
        url = f"{self.base_url}/repos/{self.repo}/actions/runs/{run_id}"
        response = self._make_request("DELETE", url)
        return response is not None and response.status_code == 204
    
    def clean_workflow_runs(self, keep_count: int = 5, workflow_filter: Optional[str] = None) -> Tuple[int, int]:
        """
        Clean up old workflow runs.
        
        Args:
            keep_count: Number of recent runs to keep per workflow
            workflow_filter: Optional workflow name to filter (e.g., 'quarto-build.yml')
            
        Returns:
            Tuple of (total_runs_found, runs_to_delete)
        """
        workflows = self.get_workflows()
        if not workflows:
            return 0, 0
            
        total_runs = 0
        total_to_delete = 0
        
        for workflow in workflows:
            workflow_name = workflow['name']
            workflow_path = workflow['path'].split('/')[-1]  # Get filename
            workflow_id = workflow['id']
            
            # Apply filter if specified
            if workflow_filter and workflow_filter not in [workflow_name, workflow_path]:
                continue
                
            print(f"\n🔍 Processing workflow: {workflow_name} ({workflow_path})")
            
            # Get all runs for this workflow
            runs = self.get_workflow_runs(workflow_id)
            total_runs += len(runs)
            
            if len(runs) <= keep_count:
                print(f"   ✅ Only {len(runs)} runs found, keeping all")
                continue
            
            # Sort runs by creation date (newest first)
            runs.sort(key=lambda x: x['created_at'], reverse=True)
            
            # Identify runs to delete (everything after keep_count)
            runs_to_keep = runs[:keep_count]
            runs_to_delete = runs[keep_count:]
            
            print(f"   📊 Total runs: {len(runs)}")
            print(f"   📌 Keeping: {len(runs_to_keep)} most recent")
            print(f"   🗑️  To delete: {len(runs_to_delete)}")
            
            if self.dry_run:
                print(f"   🔍 DRY RUN: Would delete {len(runs_to_delete)} runs")
                total_to_delete += len(runs_to_delete)
                continue
            
            # Delete old runs
            deleted_count = 0
            for run in runs_to_delete:
                run_id = run['id']
                run_number = run['run_number']
                status = run['status']
                conclusion = run['conclusion']
                created_at = run['created_at']
                
                print(f"   🗑️  Deleting run #{run_number} ({status}/{conclusion}) from {created_at}")
                
                if self.delete_workflow_run(run_id):
                    deleted_count += 1
                    # Small delay to avoid overwhelming the API
                    time.sleep(0.5)
                else:
                    print(f"   ❌ Failed to delete run #{run_number}")
            
            total_to_delete += deleted_count
            print(f"   ✅ Successfully deleted {deleted_count}/{len(runs_to_delete)} runs")
        
        return total_runs, total_to_delete
    
    def show_workflow_summary(self):
        """Show a summary of all workflows and their run counts."""
        workflows = self.get_workflows()
        if not workflows:
            return
            
        print(f"\n📊 Workflow Summary for {self.repo}")
        print("=" * 60)
        
        total_runs = 0
        for workflow in workflows:
            workflow_name = workflow['name']
            workflow_path = workflow['path'].split('/')[-1]
            workflow_id = workflow['id']
            
            runs = self.get_workflow_runs(workflow_id)
            run_count = len(runs)
            total_runs += run_count
            
            # Count by status
            statuses = {}
            for run in runs:
                status = f"{run['status']}/{run.get('conclusion', 'N/A')}"
                statuses[status] = statuses.get(status, 0) + 1
            
            print(f"{workflow_name} ({workflow_path}): {run_count} runs")
            for status, count in sorted(statuses.items()):
                print(f"  - {status}: {count}")
        
        print(f"\n📈 Total workflow runs across all workflows: {total_runs}")


def get_repo_from_git() -> Optional[str]:
    """Try to determine repository from git remote."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True,
            text=True,
            check=True
        )
        remote_url = result.stdout.strip()
        
        # Parse GitHub URL
        if 'github.com' in remote_url:
            if remote_url.startswith('git@github.com:'):
                repo = remote_url.replace('git@github.com:', '').replace('.git', '')
            elif remote_url.startswith('https://github.com/'):
                repo = remote_url.replace('https://github.com/', '').replace('.git', '')
            else:
                return None
            return repo
    except:
        return None
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Clean up old GitHub workflow runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show summary of all workflow runs
  python cleanup_workflow_runs.py --summary

  # Dry run - see what would be deleted
  python cleanup_workflow_runs.py --dry-run --keep 5

  # Clean up, keeping 10 most recent runs per workflow
  python cleanup_workflow_runs.py --keep 10

  # Clean up specific workflow only
  python cleanup_workflow_runs.py --workflow "quarto-build.yml" --keep 3

Environment Variables:
  GITHUB_TOKEN - GitHub personal access token (alternative to --token)
        """
    )
    
    parser.add_argument(
        '--token',
        help='GitHub personal access token (or set GITHUB_TOKEN env var)'
    )
    parser.add_argument(
        '--repo',
        help='Repository in format owner/repo (auto-detected from git if not provided)'
    )
    parser.add_argument(
        '--keep',
        type=int,
        default=5,
        help='Number of recent workflow runs to keep per workflow (default: 5)'
    )
    parser.add_argument(
        '--workflow',
        help='Clean specific workflow only (by name or filename)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be deleted without actually deleting'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show summary of workflow runs and exit'
    )
    
    args = parser.parse_args()
    
    # Get GitHub token
    token = args.token or os.getenv('GITHUB_TOKEN')
    if not token:
        print("❌ GitHub token required. Use --token flag or set GITHUB_TOKEN environment variable")
        print("   Generate token at: https://github.com/settings/tokens")
        print("   Required scopes: actions:write, repo")
        sys.exit(1)
    
    # Get repository
    repo = args.repo or get_repo_from_git()
    if not repo:
        print("❌ Repository not specified and could not auto-detect from git")
        print("   Use --repo owner/repo or run from a git repository")
        sys.exit(1)
    
    print(f"🚀 GitHub Workflow Cleanup for {repo}")
    print(f"   Token: {'*' * (len(token) - 4)}{token[-4:]}")
    print(f"   Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    
    # Initialize cleaner
    cleaner = GitHubWorkflowCleaner(token, repo, args.dry_run)
    
    if args.summary:
        cleaner.show_workflow_summary()
        return
    
    # Clean workflow runs
    print(f"\n🧹 Starting cleanup (keeping {args.keep} runs per workflow)")
    if args.workflow:
        print(f"   Filtering to workflow: {args.workflow}")
    
    total_runs, deleted_runs = cleaner.clean_workflow_runs(
        keep_count=args.keep,
        workflow_filter=args.workflow
    )
    
    print(f"\n📊 Cleanup Summary")
    print("=" * 40)
    print(f"Total workflow runs found: {total_runs}")
    if args.dry_run:
        print(f"Runs that would be deleted: {deleted_runs}")
        print("\n💡 Run without --dry-run to actually delete the runs")
    else:
        print(f"Runs successfully deleted: {deleted_runs}")
        print("✅ Cleanup completed!")


if __name__ == "__main__":
    main()
