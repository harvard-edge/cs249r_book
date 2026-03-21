#!/usr/bin/env python3
"""
Sync newsletter archive from Buttondown API to local markdown files.

Fetches published emails from Buttondown and writes them as .md files
in newsletter/posts/YYYY/ for Quarto's listing engine to consume.

Buttondown returns HTML bodies (from its "fancy" editor). This script:
  1. Strips HTML tags to extract a clean plain-text description
  2. Wraps the HTML body in a rawhtml block so Quarto renders it correctly
  3. Skips files that already exist (idempotent / incremental sync)

Usage:
  BUTTONDOWN_API_KEY=your-key python3 .github/scripts/sync_newsletter.py

The API key can be found at: https://buttondown.com/requests
"""

import json
import os
import re
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

import requests

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE = "https://api.buttondown.com/v1"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NEWSLETTER_DIR = REPO_ROOT / "site" / "newsletter" / "posts"
API_KEY = os.environ.get("BUTTONDOWN_API_KEY", "")


class HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and extract plain text."""

    def __init__(self):
        super().__init__()
        self._pieces = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("p", "br", "div", "li"):
            self._pieces.append(" ")

    def handle_data(self, data):
        if not self._skip:
            self._pieces.append(data)

    def get_text(self) -> str:
        text = "".join(self._pieces)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text


def strip_html(html: str) -> str:
    """Convert HTML to plain text."""
    # Remove buttondown editor comments
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    extractor = HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


def slugify(title: str) -> str:
    """Convert title to a URL-friendly slug."""
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug[:80].strip("-")


def is_html(text: str) -> bool:
    """Check if text contains HTML tags."""
    return bool(re.search(r"<[a-z][^>]*>", text, re.IGNORECASE))


def fetch_emails(api_key: str) -> list[dict]:
    """Fetch all published emails from Buttondown API."""
    emails = []
    url = f"{API_BASE}/emails?status=sent&ordering=-publish_date"
    headers = {
        "Authorization": f"Token {api_key}",
        "Accept": "application/json",
    }

    while url:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 401:
            print("Invalid API key. Get yours at https://buttondown.com/requests",
                  file=sys.stderr)
            sys.exit(1)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", data if isinstance(data, list) else [])
        emails.extend(results)

        # Handle pagination
        url = data.get("next") if isinstance(data, dict) else None

    return emails


def extract_first_image(html: str) -> str:
    """Extract the first <img> src from HTML body as fallback thumbnail."""
    match = re.search(r'<img[^>]+src="([^"]+)"', html)
    return match.group(1) if match else ""


def email_to_markdown(email: dict) -> tuple[str, str, str]:
    """Convert a Buttondown email to a Quarto-compatible markdown file.

    Returns (year, filename, content).
    """
    subject = email.get("subject", "Untitled")
    body = email.get("body", "")
    publish_date = email.get("publish_date", "")

    # Auto-categorize based on subject and body content
    subject_lower = subject.lower()
    body_lower = body.lower() if body else ""
    categories = []
    if any(w in subject_lower for w in ["spotlight", "community update", "show & tell",
                                         "show and tell", "celebration", "live event"]):
        categories.append("community")
    elif any(w in subject_lower for w in ["correction", "update", "resources",
                                           "discount", "applications", "kickoff"]):
        categories.append("update")
    elif any(w in subject_lower for w in ["interview", "prep"]):
        categories.append("interviews")
    if not categories:
        # Default: long-form content is an essay
        plain = strip_html(body) if is_html(body) else body
        if len(plain) > 1500:
            categories.append("essay")
        else:
            categories.append("update")

    # Detect guest author from body text ("Written by ..." pattern)
    author_match = re.search(
        r'[Ww]ritten\s+by\s+(?:Professor\s+)?([A-Z][a-záéíóúñü]+(?:\s+[A-Z][a-záéíóúñü]+)+)',
        strip_html(body) if is_html(body) else body
    )
    if author_match:
        author = author_match.group(1).strip()
    else:
        author = "Vijay Janapa Reddi"

    # Parse date
    if publish_date:
        try:
            dt = datetime.fromisoformat(publish_date.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
            year = dt.strftime("%Y")
        except ValueError:
            date_str = publish_date[:10]
            year = date_str[:4]
    else:
        date_str = "2026-01-01"
        year = "2026"

    slug = slugify(subject)
    filename = f"{date_str}_{slug}.md"

    # Get image: prefer Buttondown's email image, fall back to first body image
    # Skip auto-generated Buttondown placeholder images (text-on-gradient)
    image_url = email.get("image", "") or ""
    if image_url and "image-generator.buttondown.email" in image_url:
        image_url = ""  # These look bad — skip them
    # Don't fall back to body images — they're usually diagrams/figures
    # that look bad as thumbnails. Only use intentional cover images.

    # Extract clean plain-text description from body
    plain_text = strip_html(body) if is_html(body) else body
    # Take first ~200 chars, break at word boundary
    if len(plain_text) > 200:
        description = plain_text[:200].rsplit(" ", 1)[0] + "..."
    else:
        description = plain_text
    # Escape quotes for YAML
    description = description.replace('"', "'").replace("\n", " ").strip()

    # Build image line for frontmatter
    image_line = f'\nimage: "{image_url}"' if image_url else ""

    # Build the QMD content
    # If body is HTML, wrap in raw html block; otherwise keep as markdown
    if is_html(body):
        # Clean up buttondown editor comments
        clean_body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL).strip()
        body_block = f"\n```{{=html}}\n{clean_body}\n```\n"
    else:
        body_block = f"\n{body}\n"

    content = f"""---
title: "{subject.replace('"', "'")}"
date: "{date_str}"
author: "{author}"
description: "{description}"
categories: {json.dumps(categories)}{image_line}
---
{body_block}"""

    return year, filename, content


def fetch_subscriber_count(api_key: str) -> int:
    """Fetch subscriber count from Buttondown API."""
    headers = {
        "Authorization": f"Token {api_key}",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(f"{API_BASE}/subscribers", headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("count", 0)
    except Exception:
        return 0


def write_stats(email_count: int, subscriber_count: int) -> None:
    """Write stats to a file the newsletter page can read at build time."""
    stats_file = NEWSLETTER_DIR.parent / "_stats.yml"

    # Round subscriber count to nearest 50 for display
    if subscriber_count >= 100:
        display_subs = f"{(subscriber_count // 50) * 50}+"
    else:
        display_subs = str(subscriber_count)

    stats_file.write_text(
        f"# Auto-generated by sync_newsletter.py — do not edit\n"
        f"issue_count: {email_count}\n"
        f"subscriber_count: {subscriber_count}\n"
        f"subscriber_display: \"{display_subs}\"\n",
        encoding="utf-8",
    )
    print(f"  Stats: {email_count} issues, {display_subs} subscribers → {stats_file.name}")


def sync(api_key: str) -> None:
    """Main sync: fetch emails and write markdown files."""
    if not api_key:
        print("BUTTONDOWN_API_KEY not set. Set it or pass via environment.",
              file=sys.stderr)
        print("Get your API key at: https://buttondown.com/requests", file=sys.stderr)
        sys.exit(1)

    print("Fetching emails from Buttondown...")
    emails = fetch_emails(api_key)
    print(f"Found {len(emails)} published emails.")

    # Update stats
    subscriber_count = fetch_subscriber_count(api_key)
    write_stats(len(emails), subscriber_count)

    written = 0
    updated = 0
    skipped = 0
    consecutive_unchanged = 0

    for email in emails:
        year, filename, content = email_to_markdown(email)

        # Create year directory
        year_dir = NEWSLETTER_DIR / year
        year_dir.mkdir(parents=True, exist_ok=True)

        filepath = year_dir / filename

        if filepath.exists():
            # Check if Buttondown post was modified after our local copy
            mod_date = email.get("modification_date", "")
            if mod_date:
                try:
                    remote_mod = datetime.fromisoformat(
                        mod_date.replace("Z", "+00:00")
                    ).timestamp()
                    local_mod = filepath.stat().st_mtime
                    if remote_mod > local_mod:
                        # Post was updated on Buttondown — overwrite
                        filepath.write_text(content, encoding="utf-8")
                        updated += 1
                        consecutive_unchanged = 0
                        print(f"  ~ {year}/{filename} (updated)")
                        continue
                except (ValueError, OSError):
                    pass

            skipped += 1
            consecutive_unchanged += 1
            # Stop after 3 consecutive unchanged posts (newest-first order
            # means older ones won't have changed either). On first run
            # with an empty directory, this never triggers.
            if consecutive_unchanged >= 3:
                remaining = len(emails) - written - updated - skipped
                if remaining > 0:
                    print(f"  … skipping {remaining} older posts (unchanged)")
                break
            continue

        consecutive_unchanged = 0
        filepath.write_text(content, encoding="utf-8")
        written += 1
        print(f"  + {year}/{filename}")

    print(f"\nDone: {written} new, {updated} updated, "
          f"{skipped} unchanged, {len(emails)} total.")


if __name__ == "__main__":
    sync(API_KEY)
