#!/usr/bin/env python3
"""
GitHub Workflow Runs Cleanup Script using GitHub CLI

This script uses the GitHub CLI (gh) for authentication, so no separate token needed.
Just requires 'gh auth login' to be done once.

Usage:
    python3 cleanup_workflow_runs_gh.py --help
    python3 cleanup_workflow_runs_gh.py --dry-run
    python3 cleanup_workflow_runs_gh.py --keep 5
    python3 cleanup_workflow_runs_gh.py --keep 10 --workflow quarto-build.yml

Requirements:
    - GitHub CLI (gh) installed and authenticated
    - Run 'gh auth login' once to set up authentication
"""

import argparse
import json
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


class GitHubCLIWorkflowCleaner:
    """Manages cleanup of GitHub workflow runs using GitHub CLI."""
    
    def __init__(self, dry_run: bool = False):
        """
        Initialize the workflow cleaner.
        
        Args:
            dry_run: If True, only preview actions without executing
        """
        self.dry_run = dry_run
        self.repo = self._get_repo_info()
        
    def _run_gh_command(self, args: List[str]) -> Optional[Dict]:
        """Run a GitHub CLI command and return JSON result."""
        try:
            cmd = ['gh'] + args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout.strip():
                return json.loads(result.stdout)
            return {}
            
        except subprocess.CalledProcessError as e:
            if "authentication" in e.stderr.lower():
                print("❌ GitHub CLI not authenticated. Run: gh auth login")
                return None
            elif "not found" in e.stderr.lower():
                print("❌ GitHub CLI not installed. Install from: https://cli.github.com/")
                return None
            else:
                print(f"❌ GitHub CLI error: {e.stderr}")
                return None
        except json.JSONDecodeError:
            print(f"❌ Failed to parse GitHub CLI output")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    def _get_repo_info(self) -> str:
        """Get current repository info."""
        result = self._run_gh_command(['repo', 'view', '--json', 'nameWithOwner'])
        if result and 'nameWithOwner' in result:
            return result['nameWithOwner']
        
        print("❌ Could not determine repository. Make sure you're in a GitHub repository.")
        sys.exit(1)
    
    def check_auth(self) -> bool:
        """Check if GitHub CLI is authenticated."""
        try:
            result = subprocess.run(
                ['gh', 'auth', 'status'],
                capture_output=True,
                text=True,
                check=True
            )
            return "Logged in" in result.stdout
        except:
            return False
    
    def get_workflows(self) -> List[Dict]:
        """Get all workflows in the repository."""
        result = self._run_gh_command([
            'api', 
            f'/repos/{self.repo}/actions/workflows',
            '--paginate'
        ])
        
        if not result:
            return []
        
        workflows = result.get('workflows', [])
        print(f"📋 Found {len(workflows)} workflows")
        return workflows
    
    def get_workflow_runs(self, workflow_id: str) -> List[Dict]:
        """Get all runs for a specific workflow."""
        result = self._run_gh_command([
            'api',
            f'/repos/{self.repo}/actions/workflows/{workflow_id}/runs',
            '--paginate'
        ])
        
        if not result:
            return []
        
        return result.get('workflow_runs', [])
    
    def delete_workflow_run(self, run_id: str) -> bool:
        """Delete a specific workflow run."""
        if self.dry_run:
            return True
        
        try:
            subprocess.run(
                ['gh', 'api', f'/repos/{self.repo}/actions/runs/{run_id}', '-X', 'DELETE'],
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to delete run: {e.stderr}")
            return False
    
    def clean_workflow_runs(self, keep_count: int = 5, workflow_filter: Optional[str] = None) -> Tuple[int, int]:
        """
        Clean up old workflow runs.
        
        Args:
            keep_count: Number of recent runs to keep per workflow
            workflow_filter: Optional workflow filename to filter
            
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


def main():
    parser = argparse.ArgumentParser(
        description="Clean up old GitHub workflow runs using GitHub CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show summary of all workflow runs
  python3 cleanup_workflow_runs_gh.py --summary

  # Dry run - see what would be deleted
  python3 cleanup_workflow_runs_gh.py --dry-run --keep 5

  # Clean up, keeping 10 most recent runs per workflow
  python3 cleanup_workflow_runs_gh.py --keep 10

  # Clean up specific workflow only
  python3 cleanup_workflow_runs_gh.py --workflow quarto-build.yml --keep 3

Requirements:
  GitHub CLI (gh) must be installed and authenticated:
    1. Install: https://cli.github.com/
    2. Login: gh auth login
        """
    )
    
    parser.add_argument(
        '--keep',
        type=int,
        default=5,
        help='Number of recent workflow runs to keep per workflow (default: 5)'
    )
    parser.add_argument(
        '--workflow',
        help='Clean specific workflow only (by filename, e.g., quarto-build.yml)'
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
    
    # Initialize cleaner
    cleaner = GitHubCLIWorkflowCleaner(args.dry_run)
    
    # Check authentication
    if not cleaner.check_auth():
        print("❌ GitHub CLI not authenticated")
        print("   Run: gh auth login")
        print("   Make sure to grant 'workflow' permissions when prompted")
        sys.exit(1)
    
    print(f"🚀 GitHub Workflow Cleanup for {cleaner.repo}")
    print(f"   Authentication: GitHub CLI")
    print(f"   Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    
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
