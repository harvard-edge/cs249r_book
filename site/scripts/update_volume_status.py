#!/usr/bin/env python3
"""Update per-volume status badge JSON files based on GitHub Actions matrix results.

Reads job outcomes for the current workflow run and generates Shields.io endpoint
badge JSON files for each volume under `site/status/vol{1,2,3,4}.json`.
Optionally publishes the updated JSONs to the `gh-pages` branch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_DIR = REPO_ROOT / "site" / "status"

VOLUMES = [
    {"key": "vol1", "marker": "Vol I (", "label": "Vol I: Foundations"},
    {"key": "vol2", "marker": "Vol II (", "label": "Vol II: Scaling"},
    {"key": "vol3", "marker": "Vol III (", "label": "Vol III: Agentic"},
    {"key": "vol4", "marker": "Vol IV (", "label": "Vol IV: Physical AI"},
]


def fetch_workflow_jobs(repo: str, run_id: str, token: str) -> list[dict]:
    """Fetch all jobs for a given workflow run using the GitHub API."""
    jobs: list[dict] = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100&page={page}"
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "volume-status-updater",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                page_jobs = data.get("jobs", [])
                jobs.extend(page_jobs)
                if len(page_jobs) < 100:
                    break
                page += 1
        except Exception as e:
            print(f"⚠️ Warning: Could not fetch jobs from GitHub API: {e}", file=sys.stderr)
            break
    return jobs


def evaluate_volume_status(volume: dict, jobs: list[dict]) -> dict:
    """Determine the badge JSON for a volume based on its matrix build jobs."""
    vol_marker = volume["marker"]
    vol_jobs = [j for j in jobs if vol_marker in j.get("name", "")]

    badge_path = STATUS_DIR / f"{volume['key']}.json"
    existing_badge = {}
    if badge_path.exists():
        try:
            existing_badge = json.loads(badge_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not vol_jobs:
        # If no jobs ran for this volume in this run (e.g. single-volume build),
        # preserve the existing status if available, else default to passing.
        if existing_badge:
            return existing_badge
        return {
            "schemaVersion": 1,
            "label": volume["label"],
            "message": "passing",
            "color": "brightgreen",
        }

    # Check for failures
    failed = any(
        j.get("conclusion") in ["failure", "timed_out", "action_required", "stale"]
        for j in vol_jobs
    )
    if failed:
        return {
            "schemaVersion": 1,
            "label": volume["label"],
            "message": "failing",
            "color": "red",
        }

    # Check if all completed jobs succeeded
    all_succeeded = all(
        j.get("conclusion") in ["success", "skipped"]
        for j in vol_jobs
        if j.get("status") == "completed"
    )

    if all_succeeded:
        return {
            "schemaVersion": 1,
            "label": volume["label"],
            "message": "passing",
            "color": "brightgreen",
        }

    # If still in progress or inconclusive, keep existing or mark in-progress
    return existing_badge or {
        "schemaVersion": 1,
        "label": volume["label"],
        "message": "building",
        "color": "yellow",
    }


def update_status_files(jobs: list[dict]) -> None:
    """Update all volume JSON files in site/status/."""
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    for vol in VOLUMES:
        badge_data = evaluate_volume_status(vol, jobs)
        out_path = STATUS_DIR / f"{vol['key']}.json"
        out_path.write_text(json.dumps(badge_data, indent=2) + "\n", encoding="utf-8")
        print(f"📊 {vol['key']}: {badge_data['message']} ({badge_data['color']}) -> {out_path}")


def publish_to_gh_pages(repo: str, token: str) -> None:
    """Clone gh-pages and push updated status files."""
    print("🚀 Publishing per-volume status files to gh-pages...")
    repo_dir = REPO_ROOT / "_temp_gh_pages_status"
    if repo_dir.exists():
        subprocess.run(["rm", "-rf", str(repo_dir)], check=False)

    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth=1",
                "--branch=gh-pages",
                f"https://x-access-token:{token}@github.com/{repo}.git",
                str(repo_dir),
            ],
            check=True,
            capture_output=True,
        )

        dest_dir = repo_dir / "status"
        dest_dir.mkdir(parents=True, exist_ok=True)

        for vol in VOLUMES:
            src = STATUS_DIR / f"{vol['key']}.json"
            if src.exists():
                (dest_dir / f"{vol['key']}.json").write_text(
                    src.read_text(encoding="utf-8"), encoding="utf-8"
                )

        subprocess.run(["git", "config", "user.name", "github-actions[bot]"], cwd=str(repo_dir), check=True)
        subprocess.run(
            ["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"],
            cwd=str(repo_dir),
            check=True,
        )

        subprocess.run(["git", "add", "status"], cwd=str(repo_dir), check=True)

        diff_check = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=str(repo_dir), check=False
        )
        if diff_check.returncode == 0:
            print("🟡 No changes in volume status files; skipping push.")
            return

        subprocess.run(["git", "commit", "-m", "🏷️ Update per-volume status badges"], cwd=str(repo_dir), check=True)

        for attempt in range(1, 4):
            push_res = subprocess.run(["git", "push", "origin", "gh-pages"], cwd=str(repo_dir), check=False)
            if push_res.returncode == 0:
                print(f"✅ Successfully pushed volume status badges to gh-pages on attempt {attempt}")
                return
            print(f"⚠️ Push failed on attempt {attempt}, rebasing...")
            subprocess.run(["git", "pull", "--rebase", "origin", "gh-pages"], cwd=str(repo_dir), check=False)

        print("⚠️ Could not push volume status files to gh-pages after 3 attempts (non-fatal)")
    finally:
        if repo_dir.exists():
            subprocess.run(["rm", "-rf", str(repo_dir)], check=False)


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "harvard-edge/cs249r_book")
    run_id = os.environ.get("GITHUB_RUN_ID", "")

    jobs = []
    if token and run_id:
        print(f"🔎 Fetching job results for run {run_id} in {repo}...")
        jobs = fetch_workflow_jobs(repo, run_id, token)
        print(f"  Found {len(jobs)} total jobs in run.")

    update_status_files(jobs)

    if "--publish-gh-pages" in sys.argv and token and repo:
        publish_to_gh_pages(repo, token)

    return 0


if __name__ == "__main__":
    sys.exit(main())
