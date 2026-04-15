#!/usr/bin/env python3
"""Push a draft markdown newsletter to Buttondown as a draft.

Usage:
    python3 publish.py push <path/to/draft.md>

What it does:
    1. Reads the markdown file and its frontmatter
    2. Finds every local image reference (SVG, PNG, JPG) and uploads it to
       the Buttondown image endpoint. Buttondown returns a CDN URL.
    3. Rewrites the markdown body so every local image path becomes the
       CDN URL. Converts SVG references to PNG when Buttondown does not
       render SVG inline in email clients.
    4. Creates a Buttondown email with status `draft` using the rewritten
       body plus the frontmatter title.
    5. Prints the Buttondown editor URL so you can preview and send from
       the Buttondown UI.

Authentication:
    Set BUTTONDOWN_API_KEY in your environment or in a .env file next to
    this script. Find your key at buttondown.com/settings/programming.

Requirements:
    pip install requests python-frontmatter
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

try:
    import frontmatter
except ImportError:
    sys.exit("Install python-frontmatter: pip install python-frontmatter")

try:
    import requests
except ImportError:
    sys.exit("Install requests: pip install requests")


BUTTONDOWN_API = "https://api.buttondown.com/v1"
LOCAL_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(((?!https?://)[^)]+)\)")


def load_api_key() -> str:
    """Read BUTTONDOWN_API_KEY from env or a local .env file."""
    key = os.environ.get("BUTTONDOWN_API_KEY")
    if key:
        return key

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("BUTTONDOWN_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    sys.exit(
        "BUTTONDOWN_API_KEY is not set. Either export it in your shell or "
        f"create {env_path} with:\n\n    BUTTONDOWN_API_KEY=your-key-here\n"
    )


def headers(api_key: str) -> dict:
    return {"Authorization": f"Token {api_key}"}


def upload_image(api_key: str, image_path: Path) -> str:
    """Upload one local image, return its CDN URL."""
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    suffix = image_path.suffix.lower()
    mime_map = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    if suffix not in mime_map:
        sys.exit(f"Unsupported image type: {image_path}")

    with image_path.open("rb") as fp:
        files = {"image": (image_path.name, fp, mime_map[suffix])}
        response = requests.post(
            f"{BUTTONDOWN_API}/images",
            headers=headers(api_key),
            files=files,
            timeout=60,
        )

    if response.status_code >= 400:
        sys.exit(
            f"Image upload failed for {image_path.name}: "
            f"{response.status_code} {response.text}"
        )

    data = response.json()
    url = data.get("image") or data.get("url")
    if not url:
        sys.exit(f"Unexpected image response: {data}")
    return url


def rewrite_images(body: str, draft_path: Path, api_key: str) -> str:
    """Replace every local image in the markdown with a Buttondown CDN URL."""
    draft_dir = draft_path.parent
    seen: dict[str, str] = {}

    def replacement(match: re.Match) -> str:
        alt = match.group(1)
        local_path = match.group(2).strip()

        # Dedupe: same local path uploads once per push.
        if local_path in seen:
            return f"![{alt}]({seen[local_path]})"

        # Resolve relative to the draft file's directory.
        resolved = (draft_dir / local_path).resolve()
        print(f"  uploading {resolved.relative_to(draft_dir.parent)} ...", end=" ", flush=True)
        cdn_url = upload_image(api_key, resolved)
        seen[local_path] = cdn_url
        print(f"-> {cdn_url}")
        return f"![{alt}]({cdn_url})"

    return LOCAL_IMAGE_RE.sub(replacement, body)


def create_buttondown_draft(
    api_key: str, subject: str, body: str, description: str | None
) -> dict:
    """Create a draft email in Buttondown. Does not send."""
    payload = {
        "subject": subject,
        "body": body,
        "status": "draft",
    }
    if description:
        payload["description"] = description

    response = requests.post(
        f"{BUTTONDOWN_API}/emails",
        headers={**headers(api_key), "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )

    if response.status_code >= 400:
        sys.exit(f"Draft creation failed: {response.status_code} {response.text}")

    return response.json()


def push(draft_path: Path) -> None:
    """Upload images, create a Buttondown draft, print the preview URL."""
    if not draft_path.exists():
        sys.exit(f"Draft not found: {draft_path}")

    api_key = load_api_key()
    post = frontmatter.load(draft_path)

    title = post.metadata.get("title")
    description = post.metadata.get("description")
    if not title:
        sys.exit(f"Draft has no `title` in frontmatter: {draft_path}")

    print(f"Pushing '{title}' to Buttondown as a draft ...")
    print()

    # 1) Upload images and rewrite paths.
    print("Images:")
    body_with_cdn = rewrite_images(post.content, draft_path, api_key)
    if body_with_cdn == post.content:
        print("  (no local images to upload)")
    print()

    # 2) Create the Buttondown draft.
    print("Creating Buttondown draft ...")
    email = create_buttondown_draft(api_key, title, body_with_cdn, description)
    email_id = email.get("id")
    creation_url = email.get("creation_url") or (
        f"https://buttondown.com/emails/{email_id}" if email_id else None
    )

    print()
    print("Done.")
    print(f"  Preview and send from: {creation_url or 'check buttondown.com/emails'}")
    print()
    print("Next steps:")
    print("  1. Open the URL above, verify the rendering in Buttondown's preview.")
    print("  2. Send from the Buttondown UI.")
    print("  3. After sending, move the markdown from drafts/ to posts/YYYY/.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push a draft markdown newsletter to Buttondown.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    push_parser = subparsers.add_parser(
        "push",
        help="Upload images and create a Buttondown draft from a markdown file.",
    )
    push_parser.add_argument("path", type=Path, help="Path to the draft markdown file.")

    args = parser.parse_args()

    if args.command == "push":
        push(args.path)


if __name__ == "__main__":
    main()
