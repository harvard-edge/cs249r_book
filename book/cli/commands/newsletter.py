"""
Newsletter command implementation for MLSysBook CLI.

Manages newsletter drafts, publishing to Buttondown, and fetching
sent newsletters for the website archive.

Requires BUTTONDOWN_API_KEY environment variable for API operations.
"""

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

BUTTONDOWN_API_URL = "https://api.buttondown.com/v1/emails"
BUTTONDOWN_SUBSCRIBERS_URL = "https://api.buttondown.com/v1/subscribers"


class NewsletterCommand:
    """Handles newsletter operations for the MLSysBook."""

    def __init__(self, config_manager, verbose: bool = False):
        self.config_manager = config_manager
        self.verbose = verbose
        # Newsletter lives at repo root, not inside book/
        self.newsletter_dir = config_manager.root_dir / "newsletter"
        self.drafts_dir = self.newsletter_dir / "drafts"
        self.sent_dir = self.newsletter_dir / "sent"
        self.posts_dir = self.newsletter_dir / "posts"
        self.template_path = self.drafts_dir / "_template.md"

    def _get_api_key(self) -> Optional[str]:
        """Get Buttondown API key from environment."""
        key = os.environ.get("BUTTONDOWN_API_KEY")
        if not key:
            console.print("[red]BUTTONDOWN_API_KEY environment variable not set.[/red]")
            console.print("[dim]Set it with: export BUTTONDOWN_API_KEY=your-key-here[/dim]")
            console.print("[dim]Get your key at: https://buttondown.com/settings/api[/dim]")
        return key

    def _slugify(self, title: str) -> str:
        """Convert a title to a filename-safe slug."""
        slug = title.lower().strip()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_]+', '-', slug)
        slug = re.sub(r'-+', '-', slug)
        return slug.strip('-')

    def _parse_frontmatter(self, path: Path) -> dict:
        """Parse YAML front matter from a markdown file."""
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return {}
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}
        frontmatter = {}
        for line in parts[1].strip().splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                value = value.strip().strip('"').strip("'")
                frontmatter[key.strip()] = value
        return frontmatter

    def _get_body(self, path: Path) -> str:
        """Extract markdown body (everything after front matter)."""
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return text
        parts = text.split("---", 2)
        if len(parts) < 3:
            return text
        return parts[2].strip()

    def run(self, args):
        """Route newsletter subcommands."""
        if not args or args[0] in ("-h", "--help"):
            self._show_help()
            return True

        subcommand = args[0].lower()
        sub_args = args[1:]

        subcommands = {
            "new": self._handle_new,
            "list": self._handle_list,
            "preview": self._handle_preview,
            "publish": self._handle_publish,
            "fetch": self._handle_fetch,
            "status": self._handle_status,
        }

        if subcommand in subcommands:
            return subcommands[subcommand](sub_args)
        else:
            console.print(f"[red]Unknown newsletter subcommand: {subcommand}[/red]")
            self._show_help()
            return False

    def _show_help(self):
        """Display newsletter command help."""
        help_table = Table(title="Newsletter Commands", show_header=True)
        help_table.add_column("Command", style="cyan", width=36)
        help_table.add_column("Description", style="dim")
        help_table.add_row("newsletter new <title>", "Create a new draft from template")
        help_table.add_row("newsletter list", "List drafts and their status")
        help_table.add_row("newsletter preview <slug>", "Open a draft for preview")
        help_table.add_row("newsletter publish <slug>", "Push draft to Buttondown as draft email")
        help_table.add_row("newsletter fetch", "Pull sent newsletters for website archive")
        help_table.add_row("newsletter status", "Show subscriber count and recent sends")
        console.print(help_table)

    # ── new ───────────────────────────────────────────────────────────────────

    def _handle_new(self, args) -> bool:
        """Create a new newsletter draft from the template."""
        if not args:
            console.print("[red]Usage: ./binder newsletter new \"Your Newsletter Title\"[/red]")
            return False

        title = " ".join(args)
        date_str = datetime.now().strftime("%Y-%m-%d")
        slug = self._slugify(title)
        filename = f"{date_str}_{slug}.md"
        dest = self.drafts_dir / filename

        if dest.exists():
            console.print(f"[yellow]Draft already exists: {dest.name}[/yellow]")
            return False

        # Read template and populate
        if self.template_path.exists():
            template = self.template_path.read_text(encoding="utf-8")
        else:
            template = "---\ntitle: \"\"\ndate: \"\"\ndraft: true\n---\n\nWrite here.\n"

        content = template.replace("Newsletter Title Here", title)
        content = content.replace("YYYY-MM-DD", date_str)

        dest.write_text(content, encoding="utf-8")
        console.print(f"[green]Created draft: newsletter/drafts/{filename}[/green]")
        console.print(f"[dim]Edit it, then publish with: ./binder newsletter publish {slug}[/dim]")
        return True

    # ── list ──────────────────────────────────────────────────────────────────

    def _handle_list(self, args) -> bool:
        """List all newsletter drafts and sent newsletters."""
        table = Table(title="Newsletters", show_header=True)
        table.add_column("Status", style="bold", width=8)
        table.add_column("Date", width=12)
        table.add_column("Title", min_width=30)
        table.add_column("File", style="dim")

        # Drafts
        draft_files = sorted(self.drafts_dir.glob("*.md"))
        for f in draft_files:
            if f.name.startswith("_"):
                continue
            fm = self._parse_frontmatter(f)
            title = fm.get("title", f.stem)
            date = fm.get("date", "")
            is_draft = fm.get("draft", "true").lower() == "true"
            status = "[yellow]draft[/yellow]" if is_draft else "[blue]ready[/blue]"
            table.add_row(status, date, title, f"drafts/{f.name}")

        # Sent
        sent_files = sorted(self.sent_dir.glob("*.md"))
        for f in sent_files:
            fm = self._parse_frontmatter(f)
            title = fm.get("title", f.stem)
            date = fm.get("date", "")
            table.add_row("[green]sent[/green]", date, title, f"sent/{f.name}")

        if table.row_count == 0:
            console.print("[dim]No newsletters found. Create one with: ./binder newsletter new \"Title\"[/dim]")
        else:
            console.print(table)
        return True

    # ── preview ───────────────────────────────────────────────────────────────

    def _handle_preview(self, args) -> bool:
        """Find and display a draft for preview."""
        if not args:
            console.print("[red]Usage: ./binder newsletter preview <slug>[/red]")
            return False

        slug = args[0]
        draft = self._find_draft(slug)
        if not draft:
            console.print(f"[red]No draft matching '{slug}' found in newsletter/drafts/[/red]")
            return False

        fm = self._parse_frontmatter(draft)
        body = self._get_body(draft)

        panel = Panel(
            body[:2000] + ("\n..." if len(body) > 2000 else ""),
            title=fm.get("title", draft.stem),
            subtitle=fm.get("date", ""),
            border_style="cyan",
        )
        console.print(panel)
        console.print(f"[dim]Full file: {draft}[/dim]")
        return True

    # ── publish ───────────────────────────────────────────────────────────────

    def _handle_publish(self, args) -> bool:
        """Push a draft to Buttondown as a draft email (not sent)."""
        if not args:
            console.print("[red]Usage: ./binder newsletter publish <slug>[/red]")
            return False

        api_key = self._get_api_key()
        if not api_key:
            return False

        slug = args[0]
        draft = self._find_draft(slug)
        if not draft:
            console.print(f"[red]No draft matching '{slug}' found in newsletter/drafts/[/red]")
            return False

        fm = self._parse_frontmatter(draft)
        title = fm.get("title", draft.stem)
        body = self._get_body(draft)

        console.print(f"[cyan]Publishing draft to Buttondown: \"{title}\"[/cyan]")

        try:
            import urllib.request
            import urllib.error

            payload = json.dumps({
                "subject": title,
                "body": body,
                "status": "draft",
            }).encode("utf-8")

            req = urllib.request.Request(
                BUTTONDOWN_API_URL,
                data=payload,
                headers={
                    "Authorization": f"Token {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                email_id = result.get("id", "unknown")
                console.print(f"[green]Draft created in Buttondown (ID: {email_id})[/green]")
                console.print("[dim]Your wife can now review and send it from the Buttondown dashboard:[/dim]")
                console.print("[dim]https://buttondown.com/mlsysbook/emails[/dim]")
                return True

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            console.print(f"[red]Buttondown API error ({e.code}): {error_body}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Error publishing: {e}[/red]")
            return False

    # ── fetch ─────────────────────────────────────────────────────────────────

    def _handle_fetch(self, args) -> bool:
        """Fetch sent newsletters from Buttondown for the website archive."""
        api_key = self._get_api_key()
        if not api_key:
            return False

        console.print("[cyan]Fetching sent newsletters from Buttondown...[/cyan]")

        try:
            import urllib.request
            import urllib.error

            all_emails = []
            url = f"{BUTTONDOWN_API_URL}?status=sent&ordering=-publish_date"

            while url:
                req = urllib.request.Request(
                    url,
                    headers={"Authorization": f"Token {api_key}"},
                )
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                results = data.get("results", data if isinstance(data, list) else [])
                all_emails.extend(results)
                url = data.get("next") if isinstance(data, dict) else None

            if not all_emails:
                console.print("[yellow]No sent newsletters found on Buttondown.[/yellow]")
                return True

            # Write each email as a markdown file in sent/ and posts/
            count = 0
            for email in all_emails:
                subject = email.get("subject", "Untitled")
                body = email.get("body", "")
                publish_date = email.get("publish_date", "")
                email_id = email.get("id", "unknown")

                # Parse date
                if publish_date:
                    try:
                        dt = datetime.fromisoformat(publish_date.replace("Z", "+00:00"))
                        date_str = dt.strftime("%Y-%m-%d")
                    except (ValueError, TypeError):
                        date_str = publish_date[:10] if len(publish_date) >= 10 else "unknown"
                else:
                    date_str = "unknown"

                slug = self._slugify(subject)
                filename = f"{date_str}_{slug}.md"

                # Build front matter
                frontmatter = (
                    f"---\n"
                    f"title: \"{subject}\"\n"
                    f"date: \"{date_str}\"\n"
                    f"author: \"Vijay Janapa Reddi\"\n"
                    f"description: \"{subject}\"\n"
                    f"categories: [newsletter]\n"
                    f"buttondown-id: \"{email_id}\"\n"
                    f"---\n\n"
                )

                content = frontmatter + body

                # Write to both sent/ (committed) and posts/ (for Quarto listing)
                (self.sent_dir / filename).write_text(content, encoding="utf-8")
                (self.posts_dir / filename).write_text(content, encoding="utf-8")
                count += 1

            console.print(f"[green]Fetched {count} newsletter(s) to newsletter/sent/ and newsletter/posts/[/green]")
            console.print("[dim]Commit newsletter/sent/ to version-control the archive.[/dim]")
            console.print("[dim]newsletter/posts/ is gitignored (generated at build time).[/dim]")
            return True

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            console.print(f"[red]Buttondown API error ({e.code}): {error_body}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Error fetching newsletters: {e}[/red]")
            return False

    # ── status ────────────────────────────────────────────────────────────────

    def _handle_status(self, args) -> bool:
        """Show newsletter status: subscriber count, recent sends."""
        api_key = self._get_api_key()
        if not api_key:
            return False

        try:
            import urllib.request

            # Get subscriber count
            req = urllib.request.Request(
                f"{BUTTONDOWN_SUBSCRIBERS_URL}?page_size=1",
                headers={"Authorization": f"Token {api_key}"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                subscriber_count = data.get("count", "?")

            # Get recent emails
            req = urllib.request.Request(
                f"{BUTTONDOWN_API_URL}?status=sent&ordering=-publish_date&page_size=5",
                headers={"Authorization": f"Token {api_key}"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                recent = data.get("results", data if isinstance(data, list) else [])

            # Display
            console.print(Panel(
                f"[bold]{subscriber_count}[/bold] subscribers",
                title="Buttondown Status",
                border_style="green",
            ))

            if recent:
                table = Table(title="Recent Newsletters", show_header=True)
                table.add_column("Date", width=12)
                table.add_column("Subject", min_width=30)
                for email in recent[:5]:
                    subject = email.get("subject", "Untitled")
                    pub = email.get("publish_date", "")
                    date_str = pub[:10] if pub else "?"
                    table.add_row(date_str, subject)
                console.print(table)

            return True

        except Exception as e:
            console.print(f"[red]Error fetching status: {e}[/red]")
            return False

    # ── helpers ───────────────────────────────────────────────────────────────

    def _find_draft(self, slug: str) -> Optional[Path]:
        """Find a draft file matching the given slug (partial match)."""
        for f in sorted(self.drafts_dir.glob("*.md")):
            if f.name.startswith("_"):
                continue
            if slug in f.stem:
                return f
        return None
