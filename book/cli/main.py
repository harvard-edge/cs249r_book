#!/usr/bin/env python3
"""
MLSysBook CLI - Modular Entry Point

A refactored, modular command-line interface for building, previewing,
and managing the Machine Learning Systems textbook.
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.markup import escape as _rich_escape
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Import our modular components
# Ensure the book directory is in sys.path so 'from cli...' works regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.core.config import ConfigManager
from cli.core.discovery import ChapterDiscovery, AmbiguousChapterError
from cli.commands.build import BuildCommand
from cli.commands.preview import PreviewCommand
from cli.commands.doctor import DoctorCommand
from cli.commands.clean import CleanCommand
from cli.commands.maintenance import MaintenanceCommand
from cli.commands.debug import DebugCommand
from cli.commands.validate import ValidateCommand
from cli.commands.formatting import FormatCommand
from cli.commands.info import InfoCommand
from cli.commands.bib import BibCommand
from cli.commands.render import RenderCommand
from cli.commands.newsletter import NewsletterCommand
from cli.commands.headings import HeadingsCommand
from cli.commands.layout import LayoutCommand
from cli.commands.audit import AuditCommand
from cli.commands.reset import ResetCommand
from cli.commands.release import ReleaseCommand

console = Console()


def _cmd(text: str) -> str:
    """Escape command examples before rendering in Rich tables."""
    return _rich_escape(text)


class MLSysBookCLI:
    """Main CLI application class."""

    def __init__(self, verbose: bool = False, open_after: bool = False):
        """Initialize the CLI with all components.

        Args:
            verbose: If True, stream build output in real-time
            open_after: If True, open build output after successful build
        """
        self.root_dir = Path.cwd()
        self.verbose = verbose
        self.open_after = open_after

        # Initialize core components
        self.config_manager = ConfigManager(self.root_dir)
        self.chapter_discovery = ChapterDiscovery(self.config_manager.book_dir)

        # Initialize command handlers
        self.build_command = BuildCommand(self.config_manager, self.chapter_discovery, verbose=verbose, open_after=open_after)
        self.preview_command = PreviewCommand(self.config_manager, self.chapter_discovery)
        self.doctor_command = DoctorCommand(self.config_manager, self.chapter_discovery)
        self.clean_command = CleanCommand(self.config_manager, self.chapter_discovery)
        self.maintenance_command = MaintenanceCommand(self.config_manager, self.chapter_discovery)
        self.debug_command = DebugCommand(self.config_manager, self.chapter_discovery, verbose=verbose)
        self.validate_command = ValidateCommand(self.config_manager, self.chapter_discovery)
        self.format_command = FormatCommand(self.config_manager, self.chapter_discovery)
        self.info_command = InfoCommand(self.config_manager, self.chapter_discovery)
        self.bib_command = BibCommand(self.config_manager, self.chapter_discovery)
        self.render_command = RenderCommand(self.config_manager, self.chapter_discovery)
        self.newsletter_command = NewsletterCommand(self.config_manager, verbose=verbose)
        self.headings_command = HeadingsCommand(self.config_manager, self.chapter_discovery)
        self.layout_command = LayoutCommand(self.config_manager, self.chapter_discovery)
        self.audit_command = AuditCommand(self.config_manager, self.chapter_discovery)
        self.reset_command = ResetCommand(self.build_command)
        self.release_command = ReleaseCommand(self.config_manager, self.chapter_discovery)

    def show_banner(self):
        """Display the CLI banner."""
        banner = Panel(
            "[bold blue]📚 MLSysBook CLI v2.0[/bold blue]\n"
            "[dim]⚡ Modular, maintainable, and fast[/dim]",
            border_style="cyan"
        )
        console.print(banner)

    def show_help(self):
        """Display help information."""
        self.show_banner()

        # Fast Chapter Commands
        fast_table = Table(show_header=True, header_style="bold green", box=None)
        fast_table.add_column("Command", style="green", width=35)
        fast_table.add_column("Description", style="white", width=30)
        fast_table.add_column("Example", style="dim", width=30)

        fast_table.add_row(_cmd("build [fmt] [chapter[,ch2,...]]"), "Build HTML/PDF/EPUB by format", _cmd("./binder build pdf intro"))
        fast_table.add_row(_cmd("preview [chapter[,ch2,...]]"), "Start live dev server with hot reload", _cmd("./binder preview intro"))

        # Volume Commands
        vol_table = Table(show_header=True, header_style="bold magenta", box=None)
        vol_table.add_column("Command", style="magenta", width=35)
        vol_table.add_column("Description", style="white", width=30)
        vol_table.add_column("Example", style="dim", width=30)

        vol_table.add_row(_cmd("build html --vol1"), "Build Volume I website", _cmd("./binder build html --vol1"))
        vol_table.add_row(_cmd("build html --vol2"), "Build Volume II website", _cmd("./binder build html --vol2"))
        vol_table.add_row(_cmd("build pdf --vol1"), "Build Volume I as PDF", _cmd("./binder build pdf --vol1"))
        vol_table.add_row(_cmd("build pdf --vol2"), "Build Volume II as PDF", _cmd("./binder build pdf --vol2"))
        vol_table.add_row(_cmd("build pdf --vol1 --layout"), "Build Volume I PDF, then plan layout fixes", _cmd("./binder build pdf --vol1 --layout"))
        vol_table.add_row(_cmd("build epub --vol1"), "Build Volume I as EPUB", _cmd("./binder build epub --vol1"))
        vol_table.add_row(_cmd("build epub --vol2"), "Build Volume II as EPUB", _cmd("./binder build epub --vol2"))
        vol_table.add_row(_cmd("list --vol1"), "List Volume I chapters", _cmd("./binder list --vol1"))
        vol_table.add_row(_cmd("list --vol2"), "List Volume II chapters", _cmd("./binder list --vol2"))

        # Full Book Commands
        full_table = Table(show_header=True, header_style="bold blue", box=None)
        full_table.add_column("Command", style="blue", width=35)
        full_table.add_column("Description", style="white", width=30)
        full_table.add_column("Example", style="dim", width=30)

        full_table.add_row(_cmd("build"), "Build entire book as static HTML", _cmd("./binder build"))
        full_table.add_row(_cmd("build html --all"), "Build ALL chapters using HTML config", _cmd("./binder build html --all"))
        full_table.add_row(_cmd("preview"), "Start live dev server for entire book", _cmd("./binder preview"))
        full_table.add_row(_cmd("build pdf --all"), "Build full book (both volumes)", _cmd("./binder build pdf --all"))
        full_table.add_row(_cmd("build epub --all"), "Build full book (both volumes)", _cmd("./binder build epub --all"))

        # Quality Commands
        quality_table = Table(show_header=True, header_style="bold yellow", box=None)
        quality_table.add_column("Command", style="yellow", width=38)
        quality_table.add_column("Description", style="white", width=30)
        quality_table.add_column("Example", style="dim", width=28)

        quality_table.add_row(_cmd("check <group> [--scope ...]"), "Run validation checks", _cmd("./binder check refs"))
        quality_table.add_row(_cmd("check all"), "Run all validation checks", _cmd("./binder check all --vol1"))
        quality_table.add_row(_cmd("check spelling"), "Spell check prose and TikZ", _cmd("./binder check spelling"))
        quality_table.add_row(_cmd("check pdf --vol1|--vol2"), "Verify built PDF cross-refs", _cmd("./binder check pdf --vol1"))
        quality_table.add_row(_cmd("check registry"), "Registry migration gates", _cmd("./binder check registry"))
        quality_table.add_row(_cmd("audit chapter-pdf|html"), "Per-chapter build audit ledger", _cmd("./binder audit chapter-pdf --vol1 training"))
        quality_table.add_row(_cmd("release [--vol1|--vol2]"), "Run release gate and emit structured report", _cmd("./binder release --dry-run --json"))
        quality_table.add_row(_cmd("fix <topic> <action>"), "Fix/manage content", _cmd("./binder fix headers add"))
        quality_table.add_row(_cmd("format <target>"), "Auto-format content", _cmd("./binder format tables"))
        quality_table.add_row(_cmd("info stats [--by-chapter]"), "Book statistics (words, figs, ...)", _cmd("./binder info stats --vol1"))
        quality_table.add_row(_cmd("info figures [--format csv]"), "Extract figure list", _cmd("./binder info figures --vol1"))
        quality_table.add_row(_cmd("info concepts|headers|acronyms"), "Extract concepts, headers, acronyms", _cmd("./binder info concepts --vol1"))
        quality_table.add_row(_cmd("bib mechanical|normalize|sync"), "Bibliography management", _cmd("./binder bib sync --vol1"))
        quality_table.add_row(_cmd("render plots [--vol1|chapter]"), "Render matplotlib plots to PNG gallery", _cmd("./binder render plots --vol1"))
        quality_table.add_row(_cmd("layout --vol1|--vol2"), "Build/reuse PDF and emit auto-layout plan", _cmd("./binder layout --vol1 --no-build"))
        quality_table.add_row(_cmd("layout check <pdf> [--threshold]"), "Flag PDF pages with excessive bottom whitespace", _cmd("./binder layout check book.pdf"))
        quality_table.add_row(_cmd("layout tables --vol1|--vol2"), "Render table-only PDF audit/contact sheets", _cmd("./binder layout tables --vol2"))

        # Newsletter Commands
        nl_table = Table(show_header=True, header_style="bold magenta", box=None)
        nl_table.add_column("Command", style="magenta", width=38)
        nl_table.add_column("Description", style="white", width=30)
        nl_table.add_column("Example", style="dim", width=28)

        nl_table.add_row(_cmd("newsletter new <title>"), "Create a new draft", _cmd('./binder newsletter new "Vol2 Update"'))
        nl_table.add_row(_cmd("newsletter list"), "List drafts and sent", _cmd("./binder newsletter list"))
        nl_table.add_row(_cmd("newsletter preview <slug>"), "Preview a draft", _cmd("./binder newsletter preview vol2"))
        nl_table.add_row(_cmd("newsletter publish <slug>"), "Push draft to Buttondown", _cmd("./binder newsletter publish vol2"))
        nl_table.add_row(_cmd("newsletter fetch"), "Pull sent emails for website", _cmd("./binder newsletter fetch"))
        nl_table.add_row(_cmd("newsletter status"), "Subscriber count & recent", _cmd("./binder newsletter status"))

        # Management Commands
        mgmt_table = Table(show_header=True, header_style="bold blue", box=None)
        mgmt_table.add_column("Command", style="green", width=38)
        mgmt_table.add_column("Description", style="white", width=30)
        mgmt_table.add_column("Example", style="dim", width=28)

        mgmt_table.add_row(_cmd("debug <fmt> --vol1|--vol2"), "Find failing chapter + section", _cmd("./binder debug pdf --vol1"))
        mgmt_table.add_row(_cmd("reset <fmt|all> [--vol1|--vol2]"), "Reset build YAML configs", _cmd("./binder reset pdf --vol1"))
        mgmt_table.add_row(_cmd("clean [html|pdf|epub|artifacts]"), "Clean generated artifacts", _cmd("./binder clean artifacts"))
        mgmt_table.add_row(_cmd("switch <format>"), "Switch active config", _cmd("./binder switch pdf"))
        mgmt_table.add_row(_cmd("list"), "List available chapters", _cmd("./binder list"))
        mgmt_table.add_row(_cmd("status"), "Show current config status", _cmd("./binder status"))
        mgmt_table.add_row(_cmd("doctor"), "Run comprehensive health check", _cmd("./binder doctor"))
        mgmt_table.add_row(_cmd("setup"), "Setup development environment", _cmd("./binder setup"))
        mgmt_table.add_row(_cmd("help"), "Show this help", _cmd("./binder help"))

        # Display tables
        console.print(Panel(fast_table, title="⚡ Fast Chapter Commands", border_style="green"))
        console.print(Panel(vol_table, title="📖 Volume Commands", border_style="magenta"))
        console.print(Panel(full_table, title="📚 Full Book Commands", border_style="blue"))
        console.print(Panel(quality_table, title="✅ Quality & Formatting", border_style="yellow"))
        console.print(Panel(nl_table, title="📬 Newsletter", border_style="magenta"))
        console.print(Panel(mgmt_table, title="🔧 Management", border_style="cyan"))

        # Pro Tips
        examples = Text()
        examples.append("🎯 Modular CLI Examples:\n", style="bold magenta")
        examples.append("  ./binder build pdf --vol1 ", style="cyan")
        examples.append("# Build Volume I as PDF\n", style="dim")
        examples.append("  ./binder build pdf --vol2 ", style="cyan")
        examples.append("# Build Volume II as PDF\n", style="dim")
        examples.append("  ./binder build pdf vol1/intro ", style="cyan")
        examples.append("# Build specific chapter (disambiguate with vol prefix)\n", style="dim")
        examples.append("  ./binder build pdf --all ", style="cyan")
        examples.append("# Build entire book as PDF (both volumes)\n", style="dim")
        examples.append("  ./binder list --vol1 ", style="cyan")
        examples.append("# List only Volume I chapters\n", style="dim")

        console.print(Panel(examples, title="💡 Pro Tips", border_style="magenta"))

        # Global Options
        options_text = Text()
        options_text.append("🔧 Global Options:\n", style="bold yellow")
        options_text.append("  -v, --verbose  ", style="yellow")
        options_text.append("Stream build output in real-time\n", style="dim")
        options_text.append("  -o, --open     ", style="yellow")
        options_text.append("Open output after successful build\n", style="dim")
        options_text.append("  Example: ", style="dim")
        options_text.append("./binder build pdf --vol1 -v -o", style="cyan")

        console.print(Panel(options_text, title="⚙️ Options", border_style="yellow"))

    def _parse_build_args(self, args):
        """Parse `binder build` arguments into format, scope, and targets."""
        format_type = None
        volume = None
        build_all = False
        skip_hygiene = False
        skip_validate = False
        layout_after = False
        remaining = []

        for arg in args:
            lower = arg.lower()
            if lower == "--vol1":
                volume = "vol1"
            elif lower == "--vol2":
                volume = "vol2"
            elif lower == "--all":
                build_all = True
            elif lower == "--skip-hygiene":
                # Emergency bypass for the EPUB pre-render hygiene check
                # added in the fix/epub-issues work. See
                # BuildCommand._preflight_epub_hygiene for context.
                skip_hygiene = True
            elif lower == "--skip-validate":
                # Bypass the post-render smoke + epubcheck validation
                # added in the fix/epub-issues work. See
                # BuildCommand._postflight_epub_validation for context.
                # Use when iterating on a known-broken build or when
                # Java / epubcheck is unavailable locally.
                skip_validate = True
            elif lower == "--layout":
                layout_after = True
            elif format_type is None and lower in ("html", "pdf", "epub"):
                format_type = lower
            else:
                remaining.append(arg)

        if format_type is None:
            format_type = "html"

        chapters_arg = remaining[0] if remaining else None
        return (
            format_type,
            volume,
            build_all,
            chapters_arg,
            skip_hygiene,
            skip_validate,
            layout_after,
        )

    def handle_build_command(self, args):
        """Handle unified build command.

        Usage:
            ./binder build
            ./binder build pdf
            ./binder build epub --vol2
            ./binder build html intro,frameworks
        """
        if args and args[0].lower() == "reset":
            console.print("[red]`binder build reset` was removed.[/red]")
            console.print("[yellow]Use: ./binder reset <html|pdf|epub|all> [--vol1|--vol2][/yellow]")
            return False

        if "-h" in args or "--help" in args:
            console.print("Usage: ./binder build [html|pdf|epub] [chapters] [--vol1|--vol2|--all] [--skip-hygiene] [--skip-validate] [--layout]", markup=False)
            console.print("[dim]Build renders source artifacts. For PDF layout polish, add --layout to a full-volume PDF build.[/dim]")
            console.print("[dim]Examples:[/dim]")
            console.print("[dim]  ./binder build[/dim]")
            console.print("[dim]  ./binder build pdf[/dim]")
            console.print("[dim]  ./binder build pdf intro,training --vol1[/dim]")
            console.print("[dim]  ./binder build html --all[/dim]")
            console.print("[dim]  ./binder build epub --vol1[/dim]")
            console.print("[dim]  ./binder build pdf --vol1 --layout       # render Vol I, then emit auto-layout plan[/dim]")
            console.print("[dim]  ./binder build pdf --vol2 --layout       # render Vol II, then emit auto-layout plan[/dim]")
            console.print("[dim]  ./binder build epub --vol1 --skip-hygiene    # bypass pre-render hygiene check[/dim]")
            console.print("[dim]  ./binder build epub --vol1 --skip-validate   # bypass post-render validation[/dim]")
            console.print("[dim]  ./binder build pdf --vol1                  # runs pdftotext cross-ref scan after render[/dim]")
            console.print("[dim]Layout rule: --layout is accepted only for `build pdf --vol1|--vol2`; it runs the same planner as `binder layout --vol1|--vol2 --no-build`.[/dim]")
            return True

        self.config_manager.show_symlink_status()
        (
            format_type,
            volume,
            build_all,
            chapters_arg,
            skip_hygiene,
            skip_validate,
            layout_after,
        ) = self._parse_build_args(args)

        if build_all and chapters_arg:
            console.print("[red]❌ Cannot combine explicit chapters with --all[/red]")
            return False

        if layout_after and (
            format_type != "pdf" or not volume or build_all or chapters_arg
        ):
            console.print(
                "[red]❌ `--layout` is supported for full-volume PDF builds only.[/red]"
            )
            console.print(
                "[yellow]Use: ./binder build pdf --vol1 --layout "
                "or ./binder build pdf --vol2 --layout[/yellow]"
            )
            return False

        if build_all:
            if format_type == "html":
                console.print("[green]🌐 Building HTML with ALL chapters...[/green]")
                return self.build_command.build_html_only()
            console.print(f"[green]🏗️ Building entire book ({format_type.upper()})...[/green]")
            return self.build_command.build_full(format_type, skip_hygiene=skip_hygiene, skip_validate=skip_validate)

        if volume and not chapters_arg:
            volume_name = "Volume I" if volume == "vol1" else "Volume II"
            console.print(f"[magenta]🏗️ Building {volume_name} ({format_type.upper()})...[/magenta]")
            ok = self.build_command.build_volume(
                volume,
                format_type,
                skip_hygiene=skip_hygiene,
                skip_validate=skip_validate,
            )
            if ok and layout_after:
                return self.layout_command.run([f"--{volume}", "--no-build"])
            return ok

        if volume and chapters_arg:
            chapter_list = [ch.strip() for ch in chapters_arg.split(",")]
            console.print(f"[green]🏗️ Building {format_type.upper()} chapters in {volume}: {chapters_arg}[/green]")
            return self.build_command.build_chapters_with_volume(chapter_list, format_type, volume, skip_hygiene=skip_hygiene, skip_validate=skip_validate)

        if chapters_arg:
            chapter_list = [ch.strip() for ch in chapters_arg.split(",")]
            console.print(f"[green]🏗️ Building {format_type.upper()} chapter(s): {chapters_arg}[/green]")
            if format_type == "html":
                return self.build_command.build_html_only(chapter_list)
            return self.build_command.build_chapters(chapter_list, format_type, skip_hygiene=skip_hygiene, skip_validate=skip_validate)

        console.print(f"[green]🏗️ Building entire book ({format_type.upper()})...[/green]")
        if format_type == "html":
            return self.build_command.build_full("html")
        return self.build_command.build_full(format_type, skip_hygiene=skip_hygiene, skip_validate=skip_validate)

    def handle_preview_command(self, args):
        """Handle preview command."""
        if args and args[0].lower() in ("help", "-h", "--help"):
            console.print("Usage: ./binder preview [chapter]", markup=False)
            console.print("[dim]Examples:[/dim]")
            console.print("[dim]  ./binder preview[/dim]")
            console.print("[dim]  ./binder preview vol1/training[/dim]")
            return True

        self.config_manager.show_symlink_status()

        if len(args) < 1:
            # No target specified - preview entire book
            console.print("[blue]🌐 Starting preview for entire book...[/blue]")
            return self.preview_command.preview_full("html")
        else:
            # Chapter specified
            chapter = args[0]
            if ',' in chapter:
                console.print("[yellow]⚠️ Preview only supports single chapters, not multiple[/yellow]")
                console.print("[dim]💡 Use the first chapter from your list[/dim]")
                chapter = chapter.split(',')[0].strip()

            console.print(f"[blue]🌐 Starting preview for chapter: {chapter}[/blue]")
            return self.preview_command.preview_chapter(chapter)

    def handle_doctor_command(self, args):
        """Handle doctor/health check command."""
        if args and args[0].lower() in ("help", "-h", "--help"):
            console.print("Usage: ./binder doctor", markup=False)
            console.print("[dim]Run comprehensive local tooling and repository health checks.[/dim]")
            return True
        return self.doctor_command.run_health_check()

    def handle_clean_command(self, args):
        """Handle clean command."""
        if args and args[0].lower() in ("help", "-h", "--help"):
            self.clean_command.print_help()
            return True
        if len(args) > 0:
            target = args[0].lower()
            if target in ["html", "pdf", "epub"]:
                return self.clean_command.clean_format(target)
            elif target == "artifacts":
                return self.clean_command.clean_artifacts()
            else:
                console.print(f"[red]❌ Unknown clean target: {target}[/red]")
                console.print("[yellow]💡 Available: html, pdf, epub, artifacts[/yellow]")
                return False
        else:
            # Clean all
            return self.clean_command.clean_all()

    def handle_switch_command(self, args):
        """Handle switch command."""
        if args and args[0].lower() in ("help", "-h", "--help"):
            console.print("Usage: ./binder switch <html|pdf|epub>", markup=False)
            return True
        if len(args) < 1:
            console.print("[red]❌ Usage: ./binder switch <format>[/red]")
            console.print("[yellow]💡 Available formats: html, pdf, epub[/yellow]")
            return False

        format_type = args[0].lower()
        return self.maintenance_command.switch_format(format_type)

    def handle_setup_command(self, args):
        """Handle setup command."""
        if args and args[0].lower() in ("help", "-h", "--help"):
            console.print("Usage: ./binder setup", markup=False)
            console.print("[dim]Install/setup local development dependencies and pre-commit hooks.[/dim]")
            return True
        return self.maintenance_command.setup_environment()

    def handle_hello_command(self, args):
        """Handle hello command."""
        return self.maintenance_command.show_hello()

    def handle_about_command(self, args):
        """Handle about command."""
        return self.maintenance_command.show_about()

    def handle_audit_command(self, args):
        """Handle audit command group (chapter-level build audits)."""
        return self.audit_command.run(args)

    def handle_check_command(self, args):
        """Handle check (validation) command group."""
        return self.validate_command.run(args)

    def handle_fix_command(self, args):
        """Handle fix (maintenance) namespace command."""
        return self.maintenance_command.run_namespace(args)

    def handle_format_command(self, args):
        """Handle format command group."""
        return self.format_command.run(args)

    def handle_info_command(self, args):
        """Handle info command group (stats, figures)."""
        return self.info_command.run(args)

    def handle_bib_command(self, args):
        """Handle bib command group (list, clean, update, sync)."""
        return self.bib_command.run(args)

    def handle_render_command(self, args):
        """Handle render command group (plots)."""
        return self.render_command.run(args)

    def handle_newsletter_command(self, args):
        """Handle newsletter command group (new, list, preview, publish, fetch, status)."""
        return self.newsletter_command.run(args)

    def handle_headings_command(self, args):
        """Handle headings command group (check, dry-run, apply)."""
        return self.headings_command.run(args)

    def handle_layout_command(self, args):
        """Handle layout command group (check)."""
        return self.layout_command.run(args)

    def handle_reset_command(self, args):
        """Handle reset command group."""
        return self.reset_command.run(args)

    def handle_release_command(self, args):
        """Handle release orchestration."""
        return self.release_command.run(args)


    def handle_debug_command(self, args):
        """Handle debug command.

        Usage:
            ./binder debug pdf --vol1
            ./binder debug html --vol2 --chapter training
        """
        if args and args[0].lower() in ("help", "-h", "--help"):
            console.print("Usage: ./binder debug <pdf|html|epub> --vol1|--vol2 [--chapter <name>]", markup=False)
            console.print("[dim]Examples:[/dim]")
            console.print("[dim]  ./binder debug pdf --vol1[/dim]")
            console.print("[dim]  ./binder debug html --vol2 --chapter training[/dim]")
            return True

        # Parse args: first positional is format, then flags
        format_type = None
        volume = None
        chapter = None

        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--vol1":
                volume = "vol1"
            elif arg == "--vol2":
                volume = "vol2"
            elif arg == "--chapter" and i + 1 < len(args):
                i += 1
                chapter = args[i]
            elif arg in ("pdf", "html", "epub") and format_type is None:
                format_type = arg
            else:
                # Try as format or chapter
                if format_type is None and arg in ("pdf", "html", "epub"):
                    format_type = arg
                else:
                    console.print(f"[red]Unknown argument: {arg}[/red]")
                    return False
            i += 1

        if not format_type:
            format_type = "pdf"  # Default to PDF
        if not volume:
            console.print("[red]Please specify --vol1 or --vol2[/red]")
            console.print("[yellow]Usage: ./binder debug <pdf|html|epub> --vol1|--vol2 [--chapter <name>][/yellow]")
            return False

        return self.debug_command.debug_build(format_type, volume, chapter)

    def handle_list_command(self, args):
        """Handle list chapters command."""
        if args and args[0].lower() in ("help", "-h", "--help"):
            console.print("Usage: ./binder list [--vol1|--vol2]", markup=False)
            return True
        volume = None
        if len(args) > 0:
            if args[0] == "--vol1":
                volume = "vol1"
            elif args[0] == "--vol2":
                volume = "vol2"

        self.chapter_discovery.show_chapters(volume=volume)
        return True

    def handle_status_command(self, args):
        """Handle status command."""
        if args and args[0].lower() in ("help", "-h", "--help"):
            console.print("Usage: ./binder status", markup=False)
            return True
        console.print("[bold blue]📊 MLSysBook CLI Status[/bold blue]")
        console.print(f"[dim]Root directory: {self.root_dir}[/dim]")
        console.print(f"[dim]Book directory: {self.config_manager.book_dir}[/dim]")

        # Show config status
        self.config_manager.show_symlink_status()

        # Show chapter count
        chapters = self.chapter_discovery.get_all_chapters()
        console.print(f"[dim]Available chapters: {len(chapters)}[/dim]")

        return True

    def run(self, args):
        """Run the CLI with given arguments."""
        if len(args) < 1:
            self.show_help()
            return True

        command = args[0].lower()
        command_args = args[1:]

        rich_help_commands = {
            "audit", "bib", "clean", "fix", "format", "headings", "info",
            "maintain", "newsletter", "render", "reset",
        }
        if command != "check" and command not in rich_help_commands and command_args == ["help"]:
            command_args = ["--help"]

        # Command mapping
        commands = {
            "build": self.handle_build_command,
            "preview": self.handle_preview_command,
            "clean": self.handle_clean_command,
            "debug": self.handle_debug_command,
            "switch": self.handle_switch_command,
            "list": self.handle_list_command,
            "status": self.handle_status_command,
            "doctor": self.handle_doctor_command,
            "audit": self.handle_audit_command,
            "check": self.handle_check_command,
            "fix": self.handle_fix_command,
            "format": self.handle_format_command,
            "info": self.handle_info_command,
            "bib": self.handle_bib_command,
            "headings": self.handle_headings_command,
            "layout": self.handle_layout_command,
            "render": self.handle_render_command,
            "reset": self.handle_reset_command,
            "release": self.handle_release_command,
            "newsletter": self.handle_newsletter_command,
            "setup": self.handle_setup_command,
            "hello": self.handle_hello_command,
            "about": self.handle_about_command,
            # Aliases (backward compat)
            "validate": self.handle_check_command,
            "maintain": self.handle_fix_command,

            "help": lambda args: self.show_help() or True,
        }

        # Deprecated short aliases (hard-cut cleanup).
        deprecated_aliases = {
            "b": "build",
            "p": "preview",
            "l": "list",
            "s": "status",
            "d": "doctor",
            "h": "help",
        }
        if command in deprecated_aliases:
            full = deprecated_aliases[command]
            console.print(f"[red]❌ Alias '{command}' was removed.[/red]")
            console.print(f"[yellow]💡 Use: ./binder {full} ...[/yellow]")
            return False

        if command in ("html", "pdf", "epub"):
            console.print(f"[red]Top-level '{command}' commands were removed.[/red]")
            console.print(f"[yellow]Build with: ./binder build {command} ...[/yellow]")
            console.print(f"[yellow]Reset YAML with: ./binder reset {command} [--vol1|--vol2][/yellow]")
            return False

        if command in commands:
            try:
                return commands[command](command_args)
            except KeyboardInterrupt:
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                return False
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")
                return False
        else:
            console.print(f"[red]❌ Unknown command: {command}[/red]")
            console.print("[yellow]💡 Use './binder help' to see available commands[/yellow]")
            return False


def main():
    """Main entry point."""
    # Check for global flags
    args = sys.argv[1:]
    verbose = False
    open_after = False

    if "-v" in args:
        verbose = True
        args.remove("-v")
    elif "--verbose" in args:
        verbose = True
        args.remove("--verbose")

    if "-o" in args:
        open_after = True
        args.remove("-o")
    elif "--open" in args:
        open_after = True
        args.remove("--open")

    cli = MLSysBookCLI(verbose=verbose, open_after=open_after)
    success = cli.run(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
