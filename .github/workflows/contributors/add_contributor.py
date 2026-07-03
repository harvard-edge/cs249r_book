#!/usr/bin/env python3
"""Apply an All Contributors credit to one or more project configs.

Both contributor workflows use this helper so comment-triggered credits and
merge-triggered credits update the same files in the same way.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from projects import dirs as project_dirs  # noqa: E402
from projects import keys as project_keys  # noqa: E402

VALID_TYPES = {"bug", "code", "doc", "design", "ideas", "review", "test", "tool"}


def parse_list(value: str, *, valid: set[str] | None = None) -> list[str]:
    """Parse either JSON list syntax or a comma-separated string."""
    value = value.strip()
    if not value:
        return []
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("expected a JSON list")
        items = [str(item).strip().lower() for item in parsed]
    else:
        items = [part.strip().lower() for part in value.split(",")]

    out: list[str] = []
    for item in items:
        if not item:
            continue
        if valid is not None and item not in valid:
            raise ValueError(f"unknown value: {item}")
        if item not in out:
            out.append(item)
    return out


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def fetch_user(username: str) -> dict[str, str]:
    """Fetch public GitHub profile fields, with a deterministic fallback."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    request = urllib.request.Request(
        f"https://api.github.com/users/{username}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "mlsysbook-contributor-workflow",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        data = {}

    return {
        "name": data.get("name") or username,
        "avatar_url": data.get("avatar_url") or f"https://avatars.githubusercontent.com/{username}",
        "profile": data.get("html_url") or f"https://github.com/{username}",
    }


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=4) + "\n")


def add_to_config(
    config_path: Path,
    *,
    username: str,
    types: list[str],
    name: str,
    avatar_url: str,
    profile: str,
) -> bool:
    config = json.loads(config_path.read_text())
    contributors = config.setdefault("contributors", [])

    for contributor in contributors:
        if contributor.get("login", "").lower() != username.lower():
            continue

        old_types = contributor.get("contributions", [])
        new_types = sorted(set(old_types) | set(types))
        changed = new_types != old_types

        for field, value in {
            "name": name,
            "avatar_url": avatar_url,
            "profile": profile,
        }.items():
            if not contributor.get(field) and value:
                contributor[field] = value
                changed = True

        contributor["contributions"] = new_types
        if changed:
            write_json(config_path, config)
        print(f"Updated existing contributor {username} in {config_path}: {new_types}")
        return changed

    contributors.append(
        {
            "login": username,
            "name": name,
            "avatar_url": avatar_url,
            "profile": profile,
            "contributions": sorted(types),
        }
    )
    write_json(config_path, config)
    print(f"Added contributor {username} to {config_path}: {sorted(types)}")
    return True


def regenerate_readmes(root: Path, projects: list[str]) -> None:
    script_dir = Path(__file__).resolve().parent
    for project in projects:
        subprocess.run(
            [
                sys.executable,
                str(script_dir / "generate_readme_tables.py"),
                "--project",
                project,
                "--update",
            ],
            cwd=root,
            check=True,
        )
    subprocess.run(
        [sys.executable, str(script_dir / "generate_main_readme.py")],
        cwd=root,
        check=True,
    )


def set_output(name: str, value: object) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a") as f:
        f.write(f"{name}={json.dumps(value)}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--types", required=True, help="JSON list or comma-separated contribution types")
    parser.add_argument("--projects", required=True, help="JSON list or comma-separated project keys")
    parser.add_argument("--name")
    parser.add_argument("--avatar-url")
    parser.add_argument("--profile")
    parser.add_argument("--no-readme-update", action="store_true")
    args = parser.parse_args(argv)

    valid_projects = set(project_keys())
    dirs = project_dirs()
    projects = parse_list(args.projects, valid=valid_projects)
    types = parse_list(args.types, valid=VALID_TYPES)

    if not projects:
        raise SystemExit("No valid projects supplied.")
    if not types:
        raise SystemExit("No valid contribution types supplied.")

    user = fetch_user(args.username)
    name = args.name or user["name"]
    avatar_url = args.avatar_url or user["avatar_url"]
    profile = args.profile or user["profile"]

    root = repo_root()
    updated_configs: list[str] = []
    updated_dirs: list[str] = []
    changed_configs: list[str] = []
    missing: list[str] = []

    for project in projects:
        project_dir = dirs[project]
        config_path = root / project_dir / ".all-contributorsrc"
        if not config_path.is_file():
            missing.append(f"{project} ({config_path.relative_to(root)})")
            continue

        changed = add_to_config(
            config_path,
            username=args.username,
            types=types,
            name=name,
            avatar_url=avatar_url,
            profile=profile,
        )
        updated_configs.append(str(config_path.relative_to(root)))
        updated_dirs.append(project_dir)
        if changed:
            changed_configs.append(str(config_path.relative_to(root)))

    if missing:
        print("Missing contributor config(s):", ", ".join(missing), file=sys.stderr)
        return 1

    if not args.no_readme_update:
        regenerate_readmes(root, projects)

    set_output("updated_projects", projects)
    set_output("updated_dirs", updated_dirs)
    set_output("updated_configs", updated_configs)
    set_output("changed_configs", changed_configs)
    set_output("types", types)
    set_output("username", args.username)

    return 0


if __name__ == "__main__":
    sys.exit(main())
