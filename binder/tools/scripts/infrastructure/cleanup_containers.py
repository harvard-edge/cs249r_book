#!/usr/bin/env python3
"""
Container Registry Cleanup Script

This script helps clean up the GitHub Container Registry by removing
unnecessary containers and keeping only the main quarto-linux container.

Usage:
    python cleanup_containers.py
"""

import subprocess
import json
import sys
from typing import List, Dict

def run_command(cmd: List[str]) -> Dict:
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"success": True, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        return {"success": False, "stdout": e.stdout, "stderr": e.stderr, "code": e.returncode}

def get_containers() -> List[Dict]:
    """Get list of containers from GitHub API."""
    print("🔍 Fetching container list...")

    # Try to get containers via GitHub CLI
    result = run_command(["gh", "api", "orgs/harvard-edge/packages?package_type=container"])

    if not result["success"]:
        print("❌ Could not fetch containers via API")
        print("🔍 This might be due to permissions or authentication")
        return []

    try:
        containers = json.loads(result["stdout"])
        return containers
    except json.JSONDecodeError:
        print("❌ Could not parse container data")
        return []

def delete_container(container_name: str) -> bool:
    """Delete a container from the registry."""
    print(f"🗑️  Deleting container: {container_name}")

    # Use GitHub CLI to delete the container
    result = run_command([
        "gh", "api",
        f"orgs/harvard-edge/packages/container/{container_name}",
        "-X", "DELETE"
    ])

    if result["success"]:
        print(f"✅ Successfully deleted: {container_name}")
        return True
    else:
        print(f"❌ Failed to delete {container_name}: {result.get('stderr', 'Unknown error')}")
        return False

def main():
    """Main cleanup function."""
    print("🧹 Container Registry Cleanup")
    print("=" * 40)

    # Containers to keep
    keep_containers = ["quarto-linux"]

    # Get all containers
    containers = get_containers()

    if not containers:
        print("📋 Manual cleanup required:")
        print("")
        print("Please delete these containers manually from the GitHub Packages page:")
        print("1. test-permissions")
        print("2. quarto-linux/test-push")
        print("3. quarto-build-linux")
        print("4. mlsysbook-build-linux")
        print("5. linux-build")
        print("6. quarto-build")
        print("")
        print("Keep only: quarto-linux")
        print("")
        print("🔗 Go to: https://github.com/harvard-edge/cs249r_book/packages")
        return

    # Filter containers to delete
    containers_to_delete = []
    for container in containers:
        name = container.get("name", "")
        if name not in keep_containers:
            containers_to_delete.append(name)

    if not containers_to_delete:
        print("✅ No containers to delete - registry is already clean!")
        return

    print(f"📦 Found {len(containers_to_delete)} containers to delete:")
    for container in containers_to_delete:
        print(f"   - {container}")

    print(f"📦 Keeping {len(keep_containers)} containers:")
    for container in keep_containers:
        print(f"   - {container}")

    # Confirm deletion
    response = input("\n🗑️  Proceed with deletion? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Cleanup cancelled")
        return

    # Delete containers
    deleted_count = 0
    for container in containers_to_delete:
        if delete_container(container):
            deleted_count += 1

    print(f"\n✅ Cleanup complete! Deleted {deleted_count} containers.")
    print("📦 Registry now contains only the main quarto-linux container.")

if __name__ == "__main__":
    main()
