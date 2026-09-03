#!/usr/bin/env python3
"""
Container Details Lister

This script helps identify container details from workflow logs
and provides information to help distinguish between containers.
"""

import subprocess
import json
import re
from typing import List, Dict

def run_command(cmd: List[str]) -> Dict:
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"success": True, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        return {"success": False, "stdout": e.stdout, "stderr": e.stderr, "code": e.returncode}

def get_workflow_containers():
    """Get container information from workflow logs."""
    print("🔍 Analyzing workflow logs for container details...")

    # Get recent workflow runs
    result = run_command(["gh", "run", "list", "--workflow=build-linux-container.yml", "--limit", "5"])

    if not result["success"]:
        print("❌ Could not fetch workflow runs")
        return []

    workflow_runs = []
    for line in result["stdout"].split('\n'):
        if 'build-linux-container.yml' in line:
            parts = line.split()
            if len(parts) >= 4:
                run_id = parts[0]
                workflow_runs.append(run_id)

    containers = []

    for run_id in workflow_runs:
        print(f"📊 Checking workflow run: {run_id}")

        # Get workflow log
        log_result = run_command(["gh", "run", "view", "--log", run_id])

        if log_result["success"]:
            log_content = log_result["stdout"]

            # Extract container details
            container_info = extract_container_info(log_content, run_id)
            if container_info:
                containers.append(container_info)

    return containers

def extract_container_info(log_content: str, run_id: str) -> Dict:
    """Extract container information from log content."""
    info = {
        "run_id": run_id,
        "registry": None,
        "image": None,
        "tag": None,
        "size": None,
        "created_at": None,
        "description": None
    }

    # Extract registry
    registry_match = re.search(r'📊 Registry: (.+)', log_content)
    if registry_match:
        info["registry"] = registry_match.group(1)

    # Extract image
    image_match = re.search(r'📊 Image: (.+)', log_content)
    if image_match:
        info["image"] = image_match.group(1)

    # Extract tag
    tag_match = re.search(r'📊 Tag: (.+)', log_content)
    if tag_match:
        info["tag"] = tag_match.group(1)

    # Extract size
    size_match = re.search(r'📊 Size: (.+)', log_content)
    if size_match:
        info["size"] = size_match.group(1)

    # Extract creation date
    created_match = re.search(r'"org\.opencontainers\.image\.created":"([^"]+)"', log_content)
    if created_match:
        info["created_at"] = created_match.group(1)

    # Extract description
    desc_match = re.search(r'"org\.opencontainers\.image\.description":"([^"]+)"', log_content)
    if desc_match:
        info["description"] = desc_match.group(1)

    return info if any(v for v in info.values() if v) else None

def main():
    """Main function to list container details."""
    print("📦 Container Details Lister")
    print("=" * 40)

    containers = get_workflow_containers()

    if not containers:
        print("❌ No container information found in workflow logs")
        print("\n📋 Manual identification required:")
        print("1. Go to: https://github.com/orgs/harvard-edge/packages")
        print("2. Look for containers from cs249r_book repository")
        print("3. Check creation dates and sizes")
        print("4. Keep the most recent quarto-linux container")
        return

    print(f"\n📦 Found {len(containers)} container builds:")
    print("=" * 40)

    for i, container in enumerate(containers, 1):
        print(f"\n🔍 Container {i}:")
        print(f"   Run ID: {container['run_id']}")
        print(f"   Registry: {container['registry']}")
        print(f"   Image: {container['image']}")
        print(f"   Tag: {container['tag']}")
        print(f"   Size: {container['size']}")
        print(f"   Created: {container['created_at']}")
        print(f"   Description: {container['description']}")

    print("\n🎯 Recommendation:")
    print("Keep the container with:")
    print("- Size: 5.73GB")
    print("- Most recent creation date")
    print("- Description: 'Introduction to Machine Learning Systems'")
    print("- Tag: latest")

if __name__ == "__main__":
    main()
