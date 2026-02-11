"""
Maintenance commands for MLSysBook CLI.

Handles setup, switch, hello, about, and other maintenance operations.
"""

import argparse
import json
import os
import re
import subprocess
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class MaintenanceCommand:
    """Handles maintenance operations for the MLSysBook."""

    def __init__(self, config_manager, chapter_discovery):
        """Initialize maintenance command.

        Args:
            config_manager: ConfigManager instance
            chapter_discovery: ChapterDiscovery instance
        """
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery

    def switch_format(self, format_type: str) -> bool:
        """Switch active configuration format.

        Args:
            format_type: Format to switch to ('html', 'pdf', 'epub')

        Returns:
            True if switch succeeded, False otherwise
        """
        if format_type not in ["html", "pdf", "epub"]:
            console.print("[red]❌ Format must be 'html', 'pdf', or 'epub'[/red]")
            console.print("[yellow]💡 Available formats: html, pdf, epub[/yellow]")
            return False

        console.print(f"[blue]🔄 Switching to {format_type.upper()} configuration...[/blue]")

        try:
            # Set up the symlink
            config_name = self.config_manager.setup_symlink(format_type)
            console.print(f"[green]✅ Switched to {format_type.upper()} configuration[/green]")
            console.print(f"[dim]🔗 Active config: {config_name}[/dim]")

            # Show current status
            self.config_manager.show_symlink_status()

            return True

        except Exception as e:
            console.print(f"[red]❌ Error switching format: {e}[/red]")
            return False

    def show_hello(self) -> bool:
        """Show welcome message and quick start guide."""
        # Banner
        banner = Panel(
            "[bold blue]📚 Welcome to MLSysBook CLI v2.0![/bold blue]\n"
            "[dim]⚡ Modular, maintainable, and fast[/dim]\n\n"
            "[green]🎯 Ready to build amazing ML systems content![/green]",
            title="👋 Hello!",
            border_style="cyan",
            padding=(1, 2)
        )
        console.print(banner)

        # Quick start table
        quick_table = Table(show_header=True, header_style="bold green", box=None)
        quick_table.add_column("Action", style="green", width=25)
        quick_table.add_column("Command", style="cyan", width=30)
        quick_table.add_column("Description", style="dim", width=35)

        quick_table.add_row("🚀 Get started", "./binder help", "Show all available commands")
        quick_table.add_row("📋 List chapters", "./binder list", "See all available chapters")
        quick_table.add_row("🏗️ Build a chapter", "./binder build intro", "Build introduction chapter")
        quick_table.add_row("🌐 Preview live", "./binder preview intro", "Start live development server")
        quick_table.add_row("🏥 Health check", "./binder doctor", "Run comprehensive diagnostics")

        console.print(Panel(quick_table, title="🚀 Quick Start", border_style="green"))

        # Tips
        tips = Panel(
            "[bold magenta]💡 Pro Tips:[/bold magenta]\n"
            "• Use [cyan]./binder build intro,ml_systems[/cyan] to build multiple chapters\n"
            "• Use [cyan]./binder preview[/cyan] for live development with hot reload\n"
            "• Use [cyan]./binder doctor[/cyan] to check system health\n"
            "• Use [cyan]./binder clean[/cyan] to clean up build artifacts",
            title="💡 Tips",
            border_style="magenta"
        )
        console.print(tips)

        return True

    def show_about(self) -> bool:
        """Show information about the MLSysBook project."""
        # Project info
        about_panel = Panel(
            "[bold blue]📚 Machine Learning Systems Textbook[/bold blue]\n\n"
            "[white]A comprehensive textbook on engineering machine learning systems,[/white]\n"
            "[white]covering principles and practices for building AI solutions in real-world environments.[/white]\n\n"
            "[green]🎯 Author:[/green] Prof. Vijay Janapa Reddi (Harvard University)\n"
            "[green]🌐 Website:[/green] https://mlsysbook.ai\n"
            "[green]📖 Repository:[/green] https://github.com/harvard-edge/cs249r_book\n"
            "[green]⚡ CLI Version:[/green] v2.0 (Modular Architecture)",
            title="ℹ️ About MLSysBook",
            border_style="blue",
            padding=(1, 2)
        )
        console.print(about_panel)

        # Statistics
        chapters = self.chapter_discovery.get_all_chapters()
        stats_table = Table(show_header=True, header_style="bold blue", box=None)
        stats_table.add_column("Metric", style="blue", width=20)
        stats_table.add_column("Value", style="green", width=15)
        stats_table.add_column("Description", style="dim", width=35)

        stats_table.add_row("📄 Chapters", str(len(chapters)), "Total number of chapters")
        stats_table.add_row("🏗️ Formats", "3", "HTML, PDF, EPUB supported")
        stats_table.add_row("🔧 Commands", "10+", "Build, preview, maintenance")
        stats_table.add_row("🏥 Health Checks", "18", "Comprehensive diagnostics")

        console.print(Panel(stats_table, title="📊 Project Statistics", border_style="cyan"))

        # Architecture info
        arch_panel = Panel(
            "[bold magenta]🏗️ Modular CLI Architecture:[/bold magenta]\n\n"
            "[cyan]• ConfigManager:[/cyan] Handles Quarto configurations and format switching\n"
            "[cyan]• ChapterDiscovery:[/cyan] Finds and validates chapter files\n"
            "[cyan]• BuildCommand:[/cyan] Manages build operations for all formats\n"
            "[cyan]• PreviewCommand:[/cyan] Handles live development servers\n"
            "[cyan]• DoctorCommand:[/cyan] Performs comprehensive health checks\n"
            "[cyan]• CleanCommand:[/cyan] Cleans artifacts and restores configs\n"
            "[cyan]• MaintenanceCommand:[/cyan] Handles setup and maintenance tasks",
            title="🔧 Architecture",
            border_style="magenta"
        )
        console.print(arch_panel)

        return True

    def setup_environment(self) -> bool:
        """Setup development environment (simplified version)."""
        console.print("[bold blue]🔧 MLSysBook Environment Setup[/bold blue]")
        console.print("[dim]Setting up your development environment...[/dim]\n")

        # Run doctor command for comprehensive check
        console.print("[blue]🏥 Running health check first...[/blue]")

        # Import and run doctor (avoiding circular imports)
        from .doctor import DoctorCommand
        doctor = DoctorCommand(self.config_manager, self.chapter_discovery)
        health_ok = doctor.run_health_check()

        if health_ok:
            console.print("\n[green]✅ Environment setup complete![/green]")
            console.print("[dim]💡 Your system is healthy and ready for development[/dim]")
        else:
            console.print("\n[yellow]⚠️ Environment setup completed with issues[/yellow]")
            console.print("[dim]💡 Please review the health check results above[/dim]")

        # Show next steps
        next_steps = Panel(
            "[bold green]🚀 Next Steps:[/bold green]\n\n"
            "1. [cyan]./binder list[/cyan] - See all available chapters\n"
            "2. [cyan]./binder build intro[/cyan] - Build your first chapter\n"
            "3. [cyan]./binder preview intro[/cyan] - Start live development\n"
            "4. [cyan]./binder help[/cyan] - Explore all commands",
            title="🎯 Getting Started",
            border_style="green"
        )
        console.print(next_steps)

        return health_ok

    def run_namespace(self, args) -> bool:
        """Handle `binder maintain ...` namespace commands."""
        parser = argparse.ArgumentParser(
            prog="binder maintain",
            description="Maintenance namespace for non-build workflows",
            add_help=True,
        )
        parser.add_argument("topic", nargs="?", choices=["glossary", "images", "repo-health"])
        parser.add_argument("action", nargs="?")
        parser.add_argument("--vol1", action="store_true", help="Scope glossary build to vol1")
        parser.add_argument("--vol2", action="store_true", help="Scope glossary build to vol2")
        parser.add_argument("-f", "--file", action="append", default=[], help="Image file to process (repeatable)")
        parser.add_argument("--all", action="store_true", help="Process all matching images")
        parser.add_argument("--apply", action="store_true", help="Apply changes in-place")
        parser.add_argument("--quality", type=int, default=85, help="Compression quality (1-100)")
        parser.add_argument("--preserve-dimensions", action="store_true", help="Do not resize images")
        parser.add_argument("--smart-compression", action="store_true", help="Try quality first, resize only if still too large")
        parser.add_argument("--min-size-mb", type=int, default=1, help="Minimum size for --all image scan")
        parser.add_argument("--json", action="store_true", help="Emit JSON output for repo-health")

        try:
            ns = parser.parse_args(args)
        except SystemExit:
            return ("-h" in args) or ("--help" in args)

        if not ns.topic:
            parser.print_help()
            return False

        if ns.topic == "glossary":
            if ns.action not in (None, "build"):
                console.print("[red]❌ Supported action: maintain glossary build[/red]")
                return False
            volume = "vol1" if ns.vol1 and not ns.vol2 else "vol2" if ns.vol2 and not ns.vol1 else None
            return self._maintain_glossary_build(volume=volume)

        if ns.topic == "images":
            if ns.action not in (None, "compress"):
                console.print("[red]❌ Supported action: maintain images compress[/red]")
                return False
            files = list(ns.file)
            if ns.all:
                files.extend(self._find_images_for_compression(ns.min_size_mb))
                files = sorted(set(files))
            return self._maintain_images_compress(
                files=files,
                quality=ns.quality,
                apply=ns.apply,
                preserve_dimensions=ns.preserve_dimensions,
                smart_compression=ns.smart_compression,
            )

        if ns.topic == "repo-health":
            if ns.action not in (None, "check"):
                console.print("[red]❌ Supported action: maintain repo-health [check][/red]")
                return False
            return self._maintain_repo_health(min_size_mb=ns.min_size_mb, json_output=ns.json)

        return False

    def _maintain_glossary_build(self, volume: str = None) -> bool:
        """Build deduplicated volume glossary JSON files from chapter glossaries."""
        book_dir = self.config_manager.book_dir
        volumes = [volume] if volume else ["vol1", "vol2"]
        built = 0

        def standardize_term_name(term: str) -> str:
            return re.sub(r"[_\s]+", " ", term.strip().lower())

        def find_best_definition(definitions_with_chapters):
            if len(definitions_with_chapters) == 1:
                return definitions_with_chapters[0]["definition"]

            priority_chapters = ["dl_primer", "training", "ml_systems", "dnn_architectures"]
            for chapter_name in priority_chapters:
                for item in definitions_with_chapters:
                    if item["chapter"] == chapter_name and not item["definition"].startswith("Alternative definition:"):
                        return item["definition"]

            clean_definitions = []
            for item in definitions_with_chapters:
                def_text = item["definition"]
                if "Alternative definition:" in def_text:
                    def_text = def_text.split("Alternative definition:")[0].strip()
                clean_definitions.append((def_text, item["chapter"]))
            best_def, _ = max(clean_definitions, key=lambda x: len(x[0]))
            return best_def.rstrip(".")

        for vol in volumes:
            source_files = sorted((book_dir / "contents" / vol).glob("**/*_glossary.json"))
            if not source_files:
                console.print(f"[yellow]⚠️ No chapter glossary JSON files found for {vol}[/yellow]")
                continue

            chapter_data = {}
            for json_path in source_files:
                try:
                    with open(json_path, "r", encoding="utf-8") as handle:
                        data = json.load(handle)
                    chapter = data["metadata"]["chapter"]
                    chapter_data[chapter] = data["terms"]
                except Exception as exc:
                    console.print(f"[yellow]⚠️ Skipping {json_path}: {exc}[/yellow]")

            term_groups = defaultdict(list)
            for chapter, terms in chapter_data.items():
                for term_entry in terms:
                    std_name = standardize_term_name(term_entry["term"])
                    term_groups[std_name].append(
                        {
                            "original_term": term_entry["term"],
                            "definition": term_entry["definition"],
                            "chapter": chapter,
                        }
                    )

            clean_terms = []
            for _, group in sorted(term_groups.items()):
                term_names = [item["original_term"] for item in group]
                best_term_name = min(term_names, key=lambda x: (len(x), "_" in x, x.lower()))
                best_definition = find_best_definition(group)
                unique_chapters = sorted({item["chapter"] for item in group})
                chapter_source = unique_chapters[0]

                clean_term = {
                    "term": best_term_name.lower(),
                    "definition": best_definition,
                    "chapter_source": chapter_source,
                    "aliases": [],
                    "see_also": [],
                }
                if len(unique_chapters) > 1:
                    clean_term["appears_in"] = unique_chapters
                clean_terms.append(clean_term)

            clean_terms.sort(key=lambda x: x["term"])
            glossary = {
                "metadata": {
                    "type": "volume_glossary",
                    "volume": vol,
                    "version": "1.0.0",
                    "generated": datetime.now().isoformat(),
                    "total_terms": len(clean_terms),
                    "source": f"aggregated_from_{vol}_chapter_glossaries",
                    "standardized": True,
                    "description": f"Glossary for {vol.upper()} built from chapter glossaries",
                },
                "terms": clean_terms,
            }

            output_path = book_dir / "contents" / vol / "backmatter" / "glossary" / f"{vol}_glossary.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(glossary, handle, indent=2, ensure_ascii=False)
            console.print(f"[green]✅ Built {vol} glossary ({len(clean_terms)} terms): {output_path}[/green]")
            built += 1

        return built > 0

    def _find_images_for_compression(self, min_size_mb: int):
        """Find large images under contents for bulk compression."""
        contents = self.config_manager.book_dir / "contents"
        image_files = []
        min_bytes = min_size_mb * 1024 * 1024
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for image in contents.rglob(ext):
                if image.is_file() and image.stat().st_size >= min_bytes:
                    image_files.append(str(image))
        return image_files

    @staticmethod
    def _target_size_for_image(image_path: str) -> str:
        filename = os.path.basename(image_path).lower()
        if any(keyword in filename for keyword in ["setup", "kit", "board", "hardware", "assembled"]):
            return "1200x900"
        if any(keyword in filename for keyword in ["screenshot", "screen", "ui", "system"]):
            return "1000x750"
        if any(keyword in filename for keyword in ["diagram", "chart", "graph", "boat"]):
            return "800x600"
        return "1000x750"

    def _maintain_images_compress(
        self,
        files,
        quality: int = 85,
        apply: bool = False,
        preserve_dimensions: bool = False,
        smart_compression: bool = False,
    ) -> bool:
        """Compress selected images with optional in-place apply."""
        if not files:
            console.print("[yellow]⚠️ No files selected. Use -f/--file or --all[/yellow]")
            return False

        if shutil.which("magick") is None:
            console.print("[red]❌ ImageMagick `magick` command not found.[/red]")
            return False

        backup_dir = Path.cwd() / f"image_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]💾 Backup directory: {backup_dir}[/dim]")

        total_original = 0.0
        total_compressed = 0.0
        processed = 0

        for image_path in files:
            src = Path(image_path)
            if not src.exists():
                console.print(f"[yellow]⚠️ Missing file: {src}[/yellow]")
                continue

            processed += 1
            shutil.copy2(src, backup_dir / src.name)
            original_size = src.stat().st_size / (1024 * 1024)
            total_original += original_size

            quality_out = Path(f"{src}.compressed")
            resize_out = Path(f"{src}.resized")

            def run_magick(cmd):
                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.returncode == 0

            if smart_compression:
                ok = run_magick(["magick", str(src), "-quality", str(quality), "-strip", str(quality_out)])
                if not ok or not quality_out.exists():
                    console.print(f"[red]❌ Failed to compress {src}[/red]")
                    continue
                quality_size = quality_out.stat().st_size / (1024 * 1024)
                if quality_size <= 1.0:
                    out_path = quality_out
                else:
                    target_size = self._target_size_for_image(str(src))
                    ok_resize = run_magick(
                        ["magick", str(src), "-resize", f"{target_size}>", "-quality", str(quality), "-strip", str(resize_out)]
                    )
                    out_path = resize_out if ok_resize and resize_out.exists() else quality_out
            elif preserve_dimensions:
                ok = run_magick(["magick", str(src), "-quality", str(quality), "-strip", str(quality_out)])
                if not ok or not quality_out.exists():
                    console.print(f"[red]❌ Failed to compress {src}[/red]")
                    continue
                out_path = quality_out
            else:
                target_size = self._target_size_for_image(str(src))
                ok = run_magick(
                    ["magick", str(src), "-resize", f"{target_size}>", "-quality", str(quality), "-strip", str(quality_out)]
                )
                if not ok or not quality_out.exists():
                    console.print(f"[red]❌ Failed to compress {src}[/red]")
                    continue
                out_path = quality_out

            compressed_size = out_path.stat().st_size / (1024 * 1024)
            total_compressed += compressed_size
            savings = original_size - compressed_size
            savings_pct = (savings / original_size * 100) if original_size > 0 else 0
            console.print(
                f"[green]✅ {src.name}[/green] {original_size:.2f}MB -> {compressed_size:.2f}MB "
                f"(saved {savings:.2f}MB, {savings_pct:.1f}%)"
            )

            if apply:
                shutil.move(str(out_path), str(src))
                console.print(f"[dim]Applied: {src}[/dim]")
            else:
                console.print(f"[dim]Dry-run output: {out_path}[/dim]")

            # cleanup stale alternate output if unused
            for candidate in (quality_out, resize_out):
                if candidate.exists() and candidate != out_path:
                    candidate.unlink()

        if processed == 0:
            console.print("[yellow]⚠️ No valid image files were processed.[/yellow]")
            return False

        console.print(
            f"[bold]Summary:[/bold] original={total_original:.2f}MB compressed={total_compressed:.2f}MB "
            f"savings={total_original - total_compressed:.2f}MB"
        )
        if not apply:
            console.print("[dim]Use --apply to replace original files after review.[/dim]")
        return True

    def _maintain_repo_health(self, min_size_mb: int = 5, json_output: bool = False) -> bool:
        """Run repository health checks (non-destructive)."""
        repo_root = self.config_manager.root_dir

        def run(cmd):
            result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
            return result.returncode == 0, result.stdout.strip() if result.stdout else result.stderr.strip()

        ok_repo, _ = run(["git", "rev-parse", "--git-dir"])
        if not ok_repo:
            console.print("[red]❌ Not a git repository[/red]")
            return False

        stats = {}
        ok_count, count_out = run(["git", "count-objects", "-vH"])
        if ok_count:
            for line in count_out.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    stats[key.strip()] = value.strip()

        tracked_ok, tracked_out = run(["git", "ls-files"])
        tracked_files = [line for line in tracked_out.splitlines() if line] if tracked_ok else []

        min_bytes = min_size_mb * 1024 * 1024
        large_files = []
        for rel in tracked_files:
            abs_path = repo_root / rel
            if abs_path.exists() and abs_path.is_file():
                size = abs_path.stat().st_size
                if size >= min_bytes:
                    large_files.append({"path": rel, "size_mb": size / (1024 * 1024)})

        size_groups = defaultdict(list)
        for rel in tracked_files:
            abs_path = repo_root / rel
            if abs_path.exists() and abs_path.is_file():
                size = abs_path.stat().st_size
                if size > 1024:
                    size_groups[size].append(rel)
        duplicate_groups = [
            {"size_mb": size / (1024 * 1024), "count": len(paths), "files": paths}
            for size, paths in size_groups.items()
            if len(paths) > 1
        ]
        duplicate_groups.sort(key=lambda item: item["size_mb"], reverse=True)

        payload = {
            "repo": str(repo_root),
            "stats": stats,
            "large_files_count": len(large_files),
            "large_files": sorted(large_files, key=lambda x: x["size_mb"], reverse=True)[:25],
            "duplicate_groups_count": len(duplicate_groups),
            "duplicate_groups": duplicate_groups[:15],
        }

        if json_output:
            print(json.dumps(payload, indent=2))
            return True

        stat_table = Table(show_header=True, header_style="bold cyan", box=None, title="Repository Stats")
        stat_table.add_column("Metric", style="cyan")
        stat_table.add_column("Value", style="white")
        for key in ("count", "size", "in-pack", "size-pack", "packs"):
            if key in stats:
                stat_table.add_row(key, stats[key])
        console.print(stat_table)

        console.print(f"[yellow]Large tracked files >={min_size_mb}MB:[/yellow] {len(large_files)}")
        if large_files:
            large_table = Table(show_header=True, header_style="bold yellow", box=None)
            large_table.add_column("Path", style="white")
            large_table.add_column("Size (MB)", style="yellow")
            for item in sorted(large_files, key=lambda x: x["size_mb"], reverse=True)[:10]:
                large_table.add_row(item["path"], f"{item['size_mb']:.2f}")
            console.print(large_table)

        console.print(f"[yellow]Potential duplicate groups (size heuristic):[/yellow] {len(duplicate_groups)}")
        if duplicate_groups:
            dup_table = Table(show_header=True, header_style="bold magenta", box=None)
            dup_table.add_column("Size (MB)", style="magenta")
            dup_table.add_column("Count", style="white")
            dup_table.add_column("Sample Files", style="dim")
            for item in duplicate_groups[:10]:
                sample = ", ".join(item["files"][:3])
                if len(item["files"]) > 3:
                    sample += f" (+{len(item['files']) - 3} more)"
                dup_table.add_row(f"{item['size_mb']:.2f}", str(item["count"]), sample)
            console.print(dup_table)

        return True
