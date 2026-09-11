#!/usr/bin/env python3
"""
MLSysBook CLI - Modular Entry Point

A refactored, modular command-line interface for building, previewing,
and managing the Machine Learning Systems textbook.
"""

import re
import sys
import time
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


try:
    from cli.core.discovery import format_volume_display_name
except ImportError:
    from core.discovery import format_volume_display_name


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

        fast_table.add_row(_cmd("build [fmt] [chapter[,ch2,...]]"), "Build HTML/PDF/EPUB by format", _cmd("./binder/binder build pdf intro"))
        fast_table.add_row(_cmd("preview [chapter[,ch2,...]]"), "Start live dev server with hot reload", _cmd("./binder/binder preview intro"))

        # Volume Commands
        vol_table = Table(show_header=True, header_style="bold magenta", box=None)
        vol_table.add_column("Command", style="magenta", width=35)
        vol_table.add_column("Description", style="white", width=30)
        vol_table.add_column("Example", style="dim", width=30)

        vol_table.add_row(_cmd("build html --vol1"), "Build Volume I website", _cmd("./binder/binder build html --vol1"))
        vol_table.add_row(_cmd("build html --vol2"), "Build Volume II website", _cmd("./binder/binder build html --vol2"))
        vol_table.add_row(_cmd("build pdf --vol1"), "Build Volume I as PDF", _cmd("./binder/binder build pdf --vol1"))
        vol_table.add_row(_cmd("build pdf --vol2"), "Build Volume II as PDF", _cmd("./binder/binder build pdf --vol2"))
        vol_table.add_row(_cmd("build pdf --vol3"), "Build Volume III as PDF", _cmd("./binder/binder build pdf --vol3"))
        vol_table.add_row(_cmd("build pdf --vol4"), "Build Volume IV as PDF", _cmd("./binder/binder build pdf --vol4"))
        vol_table.add_row(_cmd("build pdf --vol1 --no-cover"), "Omit the designed PDF cover", _cmd("./binder/binder build pdf --vol1 --no-cover"))
        vol_table.add_row(_cmd("build pdf --vol1 --print-marks"), "Add printer camera/trim marks", _cmd("./binder/binder build pdf --vol1 --print-marks"))
        vol_table.add_row(_cmd("build pdf --vol1 --layout"), "Build Volume I PDF, then plan layout fixes", _cmd("./binder/binder build pdf --vol1 --layout"))
        vol_table.add_row(_cmd("build epub --vol1"), "Build Volume I as EPUB", _cmd("./binder/binder build epub --vol1"))
        vol_table.add_row(_cmd("build epub --vol2"), "Build Volume II as EPUB", _cmd("./binder/binder build epub --vol2"))
        vol_table.add_row(_cmd("list --vol1"), "List Volume I chapters", _cmd("./binder/binder list --vol1"))
        vol_table.add_row(_cmd("list --vol2"), "List Volume II chapters", _cmd("./binder/binder list --vol2"))

        # Full Book Commands
        full_table = Table(show_header=True, header_style="bold blue", box=None)
        full_table.add_column("Command", style="blue", width=35)
        full_table.add_column("Description", style="white", width=30)
        full_table.add_column("Example", style="dim", width=30)

        full_table.add_row(_cmd("build"), "Build entire book as static HTML", _cmd("./binder/binder build"))
        full_table.add_row(_cmd("build html --all"), "Build ALL chapters using HTML config", _cmd("./binder/binder build html --all"))
        full_table.add_row(_cmd("preview"), "Start live dev server for entire book", _cmd("./binder/binder preview"))
        full_table.add_row(_cmd("build pdf --all"), "Build full book (both volumes)", _cmd("./binder/binder build pdf --all"))
        full_table.add_row(_cmd("build epub --all"), "Build full book (both volumes)", _cmd("./binder/binder build epub --all"))

        # Quality Commands
        quality_table = Table(show_header=True, header_style="bold yellow", box=None)
        quality_table.add_column("Command", style="yellow", width=38)
        quality_table.add_column("Description", style="white", width=30)
        quality_table.add_column("Example", style="dim", width=28)

        quality_table.add_row(_cmd("check <group> [--scope ...]"), "Run validation checks", _cmd("./binder/binder check refs"))
        quality_table.add_row(_cmd("check all"), "Run all validation checks", _cmd("./binder/binder check all --vol1"))
        quality_table.add_row(_cmd("check spelling"), "Spell check prose and TikZ", _cmd("./binder/binder check spelling"))
        quality_table.add_row(_cmd("check pdf --vol1|--vol2"), "Verify built PDF cross-refs", _cmd("./binder/binder check pdf --vol1"))
        quality_table.add_row(_cmd("check registry"), "Registry migration gates", _cmd("./binder/binder check registry"))
        quality_table.add_row(_cmd("audit chapter-pdf|html"), "Per-chapter build audit ledger", _cmd("./binder/binder audit chapter-pdf --vol1 training"))
        quality_table.add_row(_cmd("release [--vol1|--vol2]"), "Run release gate and emit structured report", _cmd("./binder/binder release --dry-run --json"))
        quality_table.add_row(_cmd("fix <topic> <action>"), "Fix/manage content", _cmd("./binder/binder fix headers add"))
        quality_table.add_row(_cmd("format <target>"), "Auto-format content", _cmd("./binder/binder format tables"))
        quality_table.add_row(_cmd("info stats [--by-chapter]"), "Book statistics (words, figs, ...)", _cmd("./binder/binder info stats --vol1"))
        quality_table.add_row(_cmd("info figures [--format csv]"), "Extract figure list", _cmd("./binder/binder info figures --vol1"))
        quality_table.add_row(_cmd("info concepts|headers|acronyms"), "Extract concepts, headers, acronyms", _cmd("./binder/binder info concepts --vol1"))
        quality_table.add_row(_cmd("bib mechanical|normalize|sync"), "Bibliography management", _cmd("./binder/binder bib sync --vol1"))
        quality_table.add_row(_cmd("render plots [--vol1|chapter]"), "Render matplotlib plots to PNG gallery", _cmd("./binder/binder render plots --vol1"))
        quality_table.add_row(_cmd("layout --vol1|--vol2"), "Build/reuse PDF and emit auto-layout plan", _cmd("./binder/binder layout --vol1 --no-build"))
        quality_table.add_row(_cmd("layout chapter <name> --volN --aux <file>"), "Mapped isolated PDF component", _cmd("./binder/binder layout chapter ml_workflow --vol1 --aux full.aux"))
        quality_table.add_row(_cmd("layout check <pdf> [--threshold]"), "Flag PDF pages with excessive bottom whitespace", _cmd("./binder/binder layout check book.pdf"))
        quality_table.add_row(_cmd("layout tables --vol1|--vol2"), "Render table-only PDF audit/contact sheets", _cmd("./binder/binder layout tables --vol2"))

        # Newsletter Commands
        nl_table = Table(show_header=True, header_style="bold magenta", box=None)
        nl_table.add_column("Command", style="magenta", width=38)
        nl_table.add_column("Description", style="white", width=30)
        nl_table.add_column("Example", style="dim", width=28)

        nl_table.add_row(_cmd("newsletter new <title>"), "Create a new draft", _cmd('./binder/binder newsletter new "Vol2 Update"'))
        nl_table.add_row(_cmd("newsletter list"), "List drafts and sent", _cmd("./binder/binder newsletter list"))
        nl_table.add_row(_cmd("newsletter preview <slug>"), "Preview a draft", _cmd("./binder/binder newsletter preview vol2"))
        nl_table.add_row(_cmd("newsletter publish <slug>"), "Push draft to Buttondown", _cmd("./binder/binder newsletter publish vol2"))
        nl_table.add_row(_cmd("newsletter fetch"), "Pull sent emails for website", _cmd("./binder/binder newsletter fetch"))
        nl_table.add_row(_cmd("newsletter status"), "Subscriber count & recent", _cmd("./binder/binder newsletter status"))

        # Management Commands
        mgmt_table = Table(show_header=True, header_style="bold blue", box=None)
        mgmt_table.add_column("Command", style="green", width=38)
        mgmt_table.add_column("Description", style="white", width=30)
        mgmt_table.add_column("Example", style="dim", width=28)

        mgmt_table.add_row(_cmd("debug <fmt> --vol1|--vol2"), "Find failing chapter + section", _cmd("./binder/binder debug pdf --vol1"))
        mgmt_table.add_row(_cmd("reset <fmt|all> [--vol1|--vol2]"), "Reset build YAML configs", _cmd("./binder/binder reset pdf --vol1"))
        mgmt_table.add_row(_cmd("clean [html|pdf|epub|artifacts]"), "Clean generated artifacts", _cmd("./binder/binder clean artifacts"))
        mgmt_table.add_row(_cmd("switch <format>"), "Switch active config", _cmd("./binder/binder switch pdf"))
        mgmt_table.add_row(_cmd("list"), "List available chapters", _cmd("./binder/binder list"))
        mgmt_table.add_row(_cmd("status"), "Show current config status", _cmd("./binder/binder status"))
        mgmt_table.add_row(_cmd("doctor"), "Run comprehensive health check", _cmd("./binder/binder doctor"))
        mgmt_table.add_row(_cmd("setup"), "Setup development environment", _cmd("./binder/binder setup"))
        mgmt_table.add_row(_cmd("help"), "Show this help", _cmd("./binder/binder help"))

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
        examples.append("  ./binder/binder build pdf --vol1 ", style="cyan")
        examples.append("# Build Volume I as PDF\n", style="dim")
        examples.append("  ./binder/binder build pdf --vol2 ", style="cyan")
        examples.append("# Build Volume II as PDF\n", style="dim")
        examples.append("  ./binder/binder build pdf vol1/intro ", style="cyan")
        examples.append("# Build specific chapter (disambiguate with vol prefix)\n", style="dim")
        examples.append("  ./binder/binder build pdf --all ", style="cyan")
        examples.append("# Build entire book as PDF (both volumes)\n", style="dim")
        examples.append("  ./binder/binder list --vol1 ", style="cyan")
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
        options_text.append("./binder/binder build pdf --vol1 -v -o", style="cyan")

        console.print(Panel(options_text, title="⚙️ Options", border_style="yellow"))

    def _parse_build_args(self, args):
        """Parse `binder build` arguments into format, scope, and target components.

        Extracts output format ('html', 'pdf', 'epub'), volume selectors
        (e.g., '--vol1', '--vol4', '-1', '--tinytorch'), multi-volume flags ('--all'),
        chapter targets, and execution toggles ('--skip-hygiene', '--skip-validate',
        '--layout', '--no-cover', '--print-marks').

        Args:
            args: Raw CLI argument list.

        Returns:
            Tuple of:
                format_type (str): "html", "pdf", or "epub".
                volume (Optional[str]): Volume identifier or None.
                build_all (bool): True if full multi-volume build requested.
                chapters_arg (Optional[str]): Space/comma separated chapter specifications.
                skip_hygiene (bool): True if EPUB hygiene preflight should be bypassed.
                skip_validate (bool): True if post-render validation should be bypassed.
                layout_after (bool): True if auto-layout analysis should run after build.
                no_cover (bool): True if PDF cover page should be omitted.
                print_marks (bool): True if printer crop/trim marks should be included.

        Raises:
            ValueError: If more than one volume is explicitly specified.
        """
        format_type = None
        volume = None
        build_all = False
        skip_hygiene = False
        skip_validate = False
        layout_after = False
        no_cover = False
        print_marks = False
        remaining = []
        explicit_volumes = set()
        for arg in args:
            lower = arg.lower()
            m_vol = re.match(r"^--(vol\d+)$", lower) or re.match(r"^-(\d+)$", lower)
            if m_vol:
                val = m_vol.group(1)
                explicit_volumes.add(val if val.startswith("vol") else f"vol{val}")
            elif lower in ("--tinytorch",):
                explicit_volumes.add("tinytorch")

        if len(explicit_volumes) > 1:
            raise ValueError("Select only one volume for a build")

        for arg in args:
            lower = arg.lower()
            # With --volN, words such as "intro" and "physical" are chapter
            # queries, not legacy aliases that silently select a whole book.
            if explicit_volumes and lower in ("intro", "scaling", "agentic", "physical"):
                remaining.append(arg)
                continue
            previous_volume = volume
            m_vol = (
                re.match(r"^--(vol\d+)$", lower)
                or re.match(r"^-(\d+)$", lower)
                or re.match(r"^(vol\d+)$", lower)
                or re.match(r"^v(\d+)$", lower)
            )
            if m_vol:
                v = m_vol.group(1)
                volume = v if v.startswith("vol") else f"vol{v}"
            elif lower in ("intro",):
                volume = "vol1"
            elif lower in ("scaling",):
                volume = "vol2"
            elif lower in ("agentic",):
                volume = "vol3"
            elif lower in ("physical", "--physical"):
                volume = "vol4"
            elif lower in ("--tinytorch", "tinytorch"):
                volume = "tinytorch"
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
            elif lower == "--no-cover":
                no_cover = True
            elif lower == "--print-marks":
                print_marks = True
            elif lower == "--json":
                pass
            elif format_type is None and lower in ("html", "pdf", "epub"):
                format_type = lower
            else:
                remaining.append(arg)
            if previous_volume and volume != previous_volume:
                raise ValueError("Select only one volume for a build")

        if format_type is None:
            format_type = "html"

        chapters_arg = " ".join(remaining) if remaining else None
        return (
            format_type,
            volume,
            build_all,
            chapters_arg,
            skip_hygiene,
            skip_validate,
            layout_after,
            no_cover,
            print_marks,
        )

    def handle_build_command(self, args):
        """Handle the unified build command.

        Coordinates configuration activation, chapter resolution, rendering execution,
        and post-render validation. If ``--json`` is present, redirects diagnostic and
        progress output to stderr and emits a machine-readable JSON summary on stdout.

        Args:
            args: CLI arguments passed after 'build'.

        Returns:
            True if the build and all post-flight checks passed, False otherwise.

        Usage:
            ./binder/binder build
            ./binder/binder build pdf
            ./binder/binder build epub --vol2
            ./binder/binder build html intro,frameworks
        """
        if args and args[0].lower() == "reset":
            console.print("[red]`binder build reset` was removed.[/red]")
            console.print("[yellow]Use: ./binder/binder reset <html|pdf|epub|all> [--vol1|--vol2][/yellow]")
            return False

        if "-h" in args or "--help" in args:
            console.print("Usage: ./binder/binder build [html|pdf|epub] [chapters] [--vol1|--vol2|--vol3|--vol4|--all] [--skip-hygiene] [--skip-validate] [--layout] [--no-cover] [--print-marks] [--json]", markup=False)
            console.print("[dim]Build renders source artifacts. For PDF layout polish, add --layout to a full-volume PDF build.[/dim]")
            console.print("[dim]Examples:[/dim]")
            console.print("[dim]  ./binder/binder build[/dim]")
            console.print("[dim]  ./binder/binder build pdf[/dim]")
            console.print("[dim]  ./binder/binder build pdf intro,training --vol1[/dim]")
            console.print('[dim]  ./binder/binder build pdf "causal boundary" --vol4 --skip-validate  # isolated layout proof[/dim]')
            console.print("[dim]  ./binder/binder build html --all[/dim]")
            console.print("[dim]  ./binder/binder build epub --vol1[/dim]")
            console.print("[dim]  ./binder/binder build pdf --vol1 --layout       # render Vol I, then emit auto-layout plan[/dim]")
            console.print("[dim]  ./binder/binder build pdf --vol2 --layout       # render Vol II, then emit auto-layout plan[/dim]")
            console.print("[dim]  ./binder/binder build pdf --vol1               # include the cover; omit printer marks[/dim]")
            console.print("[dim]  ./binder/binder build pdf --vol1 --no-cover    # omit the designed cover[/dim]")
            console.print("[dim]  ./binder/binder build pdf --vol1 --print-marks # add printer camera/trim marks[/dim]")
            console.print("[dim]  ./binder/binder build epub --vol1 --skip-hygiene    # bypass pre-render hygiene check[/dim]")
            console.print("[dim]  ./binder/binder build epub --vol1 --skip-validate   # bypass post-render validation[/dim]")
            console.print("[dim]  ./binder/binder build pdf --vol1                  # runs pdftotext cross-ref scan after render[/dim]")
            console.print("[dim]  ./binder/binder build html intro --vol1 --json    # machine-readable build summary[/dim]")
            console.print("[dim]Layout rule: --layout is accepted only for `build pdf --vol1|--vol2`; it runs the same planner as `binder layout --vol1|--vol2 --no-build`.[/dim]")
            return True

        json_output = any(a.lower() == "--json" for a in args) if args else False
        status_console = Console(stderr=True) if json_output else console

        def _emit_json(
            success: bool,
            fmt=None,
            vol=None,
            b_all=False,
            chs=None,
            elapsed=0.0,
            err_msg=None,
        ):
            """Emit a pure JSON summary to stdout without soft-wrap distortion."""
            import json as _json
            result_payload = {
                "success": bool(success),
                "format": fmt,
                "volume": vol,
                "all": b_all,
                "chapters": chs,
                "elapsed_seconds": round(elapsed, 2),
                "log_path": str(getattr(self.build_command, "_last_build_log", "")) or None,
            }
            if err_msg:
                result_payload["error"] = err_msg
            json_text = _json.dumps(result_payload, indent=2)
            if console.file not in (sys.stdout, sys.__stdout__, sys.stderr, sys.__stderr__):
                console.print(json_text, soft_wrap=True, highlight=False)
            else:
                sys.stdout.write(json_text + "\n")
                sys.stdout.flush()

        if not json_output:
            self.config_manager.show_active_config()

        format_type = None
        volume = None
        build_all = False
        chapters_arg = None
        skip_hygiene = False
        skip_validate = False
        layout_after = False
        no_cover = False
        print_marks = False

        try:
            (
                format_type,
                volume,
                build_all,
                chapters_arg,
                skip_hygiene,
                skip_validate,
                layout_after,
                no_cover,
                print_marks,
            ) = self._parse_build_args(args)
        except Exception as e:
            status_console.print(f"[red]❌ Error: {e}[/red]")
            if json_output:
                _emit_json(False, fmt=format_type, vol=volume, b_all=build_all, err_msg=str(e))
            return False

        if build_all and chapters_arg:
            status_console.print("[red]❌ Cannot combine explicit chapters with --all[/red]")
            if json_output:
                _emit_json(
                    False,
                    fmt=format_type,
                    vol=volume,
                    b_all=True,
                    chs=[ch.strip() for ch in chapters_arg.split(",")],
                    err_msg="Cannot combine explicit chapters with --all",
                )
            return False

        if layout_after and (
            format_type != "pdf" or not volume or build_all or chapters_arg
        ):
            status_console.print(
                "[red]❌ `--layout` is supported for full-volume PDF builds only.[/red]"
            )
            status_console.print(
                "[yellow]Use: ./binder/binder build pdf --vol1 --layout "
                "or ./binder/binder build pdf --vol2 --layout[/yellow]"
            )
            if json_output:
                _emit_json(
                    False,
                    fmt=format_type,
                    vol=volume,
                    b_all=build_all,
                    err_msg="`--layout` is supported for full-volume PDF builds only",
                )
            return False

        if no_cover and (
            format_type != "pdf" or not volume or build_all or chapters_arg
        ):
            status_console.print(
                "[yellow]⚠️ `--no-cover` is honored only for full-volume PDF builds "
                "(or --vol2, --vol3, --vol4).[/yellow]"
            )

        if print_marks and (
            format_type != "pdf" or not volume or build_all or chapters_arg
        ):
            status_console.print(
                "[yellow]⚠️ `--print-marks` is honored only for full-volume PDF builds "
                "(or --vol2, --vol3, --vol4).[/yellow]"
            )

        t0 = time.time()
        ok = False

        from contextlib import redirect_stdout, nullcontext
        stdout_redirect = redirect_stdout(sys.stderr) if json_output else nullcontext()

        try:
            with stdout_redirect:
                if build_all:
                    if format_type == "html":
                        status_console.print("[green]🌐 Building HTML with ALL chapters...[/green]")
                        ok = self.build_command.build_html_only()
                    else:
                        status_console.print(f"[green]🏗️ Building entire book ({format_type.upper()})...[/green]")
                        ok = self.build_command.build_full(format_type, skip_hygiene=skip_hygiene, skip_validate=skip_validate)
                elif volume and not chapters_arg:
                    volume_name = format_volume_display_name(volume)
                    status_console.print(f"[magenta]🏗️ Building {volume_name} ({format_type.upper()})...[/magenta]")
                    ok = self.build_command.build_volume(
                        volume,
                        format_type,
                        skip_hygiene=skip_hygiene,
                        skip_validate=skip_validate,
                        no_cover=no_cover,
                        print_marks=print_marks,
                    )
                    if ok and layout_after:
                        ok = self.layout_command.run([f"--{volume}", "--no-build"])
                elif volume and chapters_arg:
                    chapter_list = [ch.strip() for ch in chapters_arg.split(",")]
                    status_console.print(f"[green]🏗️ Building {format_type.upper()} chapters in {volume}: {chapters_arg}[/green]")
                    ok = self.build_command.build_chapters_with_volume(chapter_list, format_type, volume, skip_hygiene=skip_hygiene, skip_validate=skip_validate)
                elif chapters_arg:
                    chapter_list = [ch.strip() for ch in chapters_arg.split(",")]
                    status_console.print(f"[green]🏗️ Building {format_type.upper()} chapter(s): {chapters_arg}[/green]")
                    ok = self.build_command.build_chapters(chapter_list, format_type, skip_hygiene=skip_hygiene, skip_validate=skip_validate)
                else:
                    status_console.print(f"[green]🏗️ Building entire book ({format_type.upper()})...[/green]")
                    if format_type == "html":
                        ok = self.build_command.build_full("html")
                    else:
                        ok = self.build_command.build_full(format_type, skip_hygiene=skip_hygiene, skip_validate=skip_validate)
        except Exception as e:
            status_console.print(f"[red]❌ Build error: {e}[/red]")
            if json_output:
                _emit_json(
                    False,
                    fmt=format_type,
                    vol=volume,
                    b_all=build_all,
                    chs=[ch.strip() for ch in chapters_arg.split(",")] if chapters_arg else None,
                    elapsed=time.time() - t0,
                    err_msg=str(e),
                )
            return False

        if json_output:
            _emit_json(
                ok,
                fmt=format_type,
                vol=volume,
                b_all=build_all,
                chs=[ch.strip() for ch in chapters_arg.split(",")] if chapters_arg else None,
                elapsed=time.time() - t0,
            )

        return ok

    def handle_preview_command(self, args):
        """Handle preview command to launch Quarto live dev server with hot reload.

        Args:
            args: CLI arguments (optional single chapter specification).

        Returns:
            True if dev server exited cleanly, False on error.
        """
        if args and args[0].lower() in ("help", "-h", "--help"):
            console.print("Usage: ./binder/binder preview [chapter]", markup=False)
            console.print("[dim]Examples:[/dim]")
            console.print("[dim]  ./binder/binder preview[/dim]")
            console.print("[dim]  ./binder/binder preview vol1/training[/dim]")
            return True

        self.config_manager.show_active_config()

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
        """Handle doctor health check command to verify dependencies and configuration.

        Args:
            args: CLI arguments.

        Returns:
            True if all required health checks passed, False otherwise.
        """
        if args and args[0].lower() in ("help", "-h", "--help"):
            console.print("Usage: ./binder/binder doctor", markup=False)
            console.print("[dim]Run comprehensive local tooling and repository health checks.[/dim]")
            return True
        return self.doctor_command.run_health_check()

    def handle_clean_command(self, args):
        """Handle clean command to remove generated build artifacts and caches.

        Args:
            args: Target format or scope ('html', 'pdf', 'epub', 'artifacts', or empty for all).

        Returns:
            True on successful cleanup, False on error.
        """
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
        """Handle switch command to swap active _quarto.yml and index.qmd configuration.

        Args:
            args: Target format ('html', 'pdf', 'epub').

        Returns:
            True on successful switch, False on error.
        """
        if args and args[0].lower() in ("help", "-h", "--help"):
            console.print("Usage: ./binder/binder switch <html|pdf|epub>", markup=False)
            return True
        if len(args) < 1:
            console.print("[red]❌ Usage: ./binder/binder switch <format>[/red]")
            console.print("[yellow]💡 Available formats: html, pdf, epub[/yellow]")
            return False

        format_type = args[0].lower()
        return self.maintenance_command.switch_format(format_type)

    def handle_setup_command(self, args):
        """Handle setup command to configure development environment and hooks.

        Args:
            args: CLI arguments.

        Returns:
            True on success, False on error.
        """
        if args and args[0].lower() in ("help", "-h", "--help"):
            console.print("Usage: ./binder/binder setup", markup=False)
            console.print("[dim]Install/setup local development dependencies and pre-commit hooks.[/dim]")
            return True
        return self.maintenance_command.setup_environment()

    def handle_hello_command(self, args):
        """Handle hello command.

        Args:
            args: CLI arguments.

        Returns:
            True.
        """
        return self.maintenance_command.show_hello()

    def handle_about_command(self, args):
        """Handle about command.

        Args:
            args: CLI arguments.

        Returns:
            True.
        """
        return self.maintenance_command.show_about()

    def handle_audit_command(self, args):
        """Handle audit command group for chapter-level build audits.

        Args:
            args: CLI arguments passed to AuditCommand.

        Returns:
            True if audit checks passed, False otherwise.
        """
        return self.audit_command.run(args)

    def handle_check_command(self, args):
        """Handle check (validation) command group.

        Args:
            args: CLI arguments passed to ValidateCommand.

        Returns:
            True if validation checks passed, False otherwise.
        """
        return self.validate_command.run(args)

    def handle_fix_command(self, args):
        """Handle fix (maintenance) namespace command.

        Args:
            args: CLI arguments passed to MaintenanceCommand.

        Returns:
            True if fix operations succeeded, False otherwise.
        """
        return self.maintenance_command.run_namespace(args)

    def handle_format_command(self, args):
        """Handle format command group for automated content formatting.

        Args:
            args: CLI arguments passed to FormatCommand.

        Returns:
            True on success, False on error.
        """
        return self.format_command.run(args)

    def handle_info_command(self, args):
        """Handle info command group (statistics, figures, concepts, acronyms).

        Args:
            args: CLI arguments passed to InfoCommand.

        Returns:
            True on success, False on error.
        """
        return self.info_command.run(args)

    def handle_bib_command(self, args):
        """Handle bib command group for bibliography management and verification.

        Args:
            args: CLI arguments passed to BibCommand.

        Returns:
            True on success, False on error.
        """
        return self.bib_command.run(args)

    def handle_render_command(self, args):
        """Handle render command group for figure and plot generation.

        Args:
            args: CLI arguments passed to RenderCommand.

        Returns:
            True on success, False on error.
        """
        return self.render_command.run(args)

    def handle_newsletter_command(self, args):
        """Handle newsletter command group (creation, preview, publication).

        Args:
            args: CLI arguments passed to NewsletterCommand.

        Returns:
            True on success, False on error.
        """
        return self.newsletter_command.run(args)

    def handle_headings_command(self, args):
        """Handle headings command group for header formatting and capitalization.

        Args:
            args: CLI arguments passed to HeadingsCommand.

        Returns:
            True on success, False on error.
        """
        return self.headings_command.run(args)

    def handle_layout_command(self, args):
        """Handle layout command group for PDF layout analysis and geometry checks.

        Args:
            args: CLI arguments passed to LayoutCommand.

        Returns:
            True on success, False on error.
        """
        return self.layout_command.run(args)

    def handle_reset_command(self, args):
        """Handle reset command group to restore pristine YAML configuration files.

        Args:
            args: CLI arguments passed to ResetCommand.

        Returns:
            True on success, False on error.
        """
        return self.reset_command.run(args)

    def handle_release_command(self, args):
        """Handle release orchestration and release gate execution.

        Args:
            args: CLI arguments passed to ReleaseCommand.

        Returns:
            True on successful release verification, False otherwise.
        """
        return self.release_command.run(args)


    def handle_debug_command(self, args):
        """Handle debug command for pinpointing failing chapters and sections.

        Args:
            args: CLI arguments specifying format, volume, and optional chapter.

        Returns:
            True if debug run completed successfully, False otherwise.

        Usage:
            ./binder/binder debug pdf --vol1
            ./binder/binder debug html --vol2 --chapter training
        """
        if args and args[0].lower() in ("help", "-h", "--help"):
            console.print("Usage: ./binder/binder debug <pdf|html|epub> --vol1|--vol2 [--chapter <name>]", markup=False)
            console.print("[dim]Examples:[/dim]")
            console.print("[dim]  ./binder/binder debug pdf --vol1[/dim]")
            console.print("[dim]  ./binder/binder debug html --vol2 --chapter training[/dim]")
            return True

        # Parse args: first positional is format, then flags
        format_type = None
        volume = None
        chapter = None

        i = 0
        while i < len(args):
            arg = args[i]
            m_vol = re.match(r"^--(vol\d+)$", arg.lower()) or re.match(r"^-(vol\d+)$", arg.lower())
            if m_vol:
                volume = m_vol.group(1)
            elif arg.lower() in ("--tinytorch", "tinytorch"):
                volume = "tinytorch"
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
            console.print("[red]Please specify a volume (e.g. --vol1, --vol2)[/red]")
            console.print("[yellow]Usage: ./binder/binder debug <pdf|html|epub> --volN [--chapter <name>][/yellow]")
            return False

        return self.debug_command.debug_build(format_type, volume, chapter)

    def handle_list_command(self, args):
        """Handle list chapters command to display discovered chapters and metadata.

        Args:
            args: CLI arguments specifying optional volume filter.

        Returns:
            True.
        """
        if args and args[0].lower() in ("help", "-h", "--help"):
            console.print("Usage: ./binder/binder list [--vol1|--vol2|...]", markup=False)
            return True
        volume = None
        if len(args) > 0:
            lower = args[0].lower()
            m_vol = re.match(r"^--(vol\d+)$", lower) or re.match(r"^(vol\d+)$", lower)
            if m_vol:
                volume = m_vol.group(1)
            elif lower in ("--tinytorch", "tinytorch"):
                volume = "tinytorch"

        self.chapter_discovery.show_chapters(volume=volume)
        return True

    def handle_status_command(self, args):
        """Handle status command to display current active configuration and index state.

        Args:
            args: CLI arguments.

        Returns:
            True.
        """
        if args and args[0].lower() in ("help", "-h", "--help"):
            console.print("Usage: ./binder/binder status", markup=False)
            return True
        console.print("[bold blue]📊 MLSysBook CLI Status[/bold blue]")
        console.print(f"[dim]Root directory: {self.root_dir}[/dim]")
        console.print(f"[dim]Book directory: {self.config_manager.book_dir}[/dim]")

        # Show config status
        self.config_manager.show_active_config()

        # Show chapter count
        chapters = self.chapter_discovery.get_all_chapters()
        console.print(f"[dim]Available chapters: {len(chapters)}[/dim]")

        return True

    def run(self, args):
        """Dispatch CLI command and arguments to the appropriate handler.

        Args:
            args: Full command line arguments excluding the script name.

        Returns:
            True if the executed command succeeded, False on failure or error.
        """
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
            console.print(f"[yellow]💡 Use: ./binder/binder {full} ...[/yellow]")
            return False

        if command in ("html", "pdf", "epub"):
            console.print(f"[red]Top-level '{command}' commands were removed.[/red]")
            console.print(f"[yellow]Build with: ./binder/binder build {command} ...[/yellow]")
            console.print(f"[yellow]Reset YAML with: ./binder/binder reset {command} [--vol1|--vol2][/yellow]")
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
            console.print("[yellow]💡 Use './binder/binder help' to see available commands[/yellow]")
            return False


def main():
    """Main CLI entry point: parses global verbose and open flags and invokes MLSysBookCLI."""
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
