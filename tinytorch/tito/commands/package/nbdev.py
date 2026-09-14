"""
nbdev command for TinyTorch CLI: runs nbdev commands for notebook development.
"""

import subprocess
from argparse import ArgumentParser, Namespace
from rich.panel import Panel

from ..base import BaseCommand

class NbdevCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "nbdev"

    @property
    def description(self) -> str:
        return "nbdev notebook development commands"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--export", action="store_true", help="Export notebooks to Python package")
        parser.add_argument("--build-docs", action="store_true", help="Build documentation from notebooks")
        parser.add_argument("--test", action="store_true", help="Run notebook tests")
        parser.add_argument("--clean", action="store_true", help="Clean notebook outputs")
        parser.add_argument("--all", action="store_true", help="Export all modules (use with --export)")
        parser.add_argument("module", nargs="?", help="Export specific module (use with --export)")

    def run(self, args: Namespace) -> int:
        console = self.console

        console.print(Panel("📓 nbdev Notebook Development",
                           title="Notebook Tools", border_style="bright_cyan"))

        if args.export:
            # Use the export command logic
            from ..dev.export import DevExportCommand
            export_cmd = DevExportCommand(self.config)
            export_args = ArgumentParser()
            export_cmd.add_arguments(export_args)

            # Build the arguments for export command
            export_arg_list = []
            if args.all:
                export_arg_list.append("--all")
            elif args.module:
                export_arg_list.append(args.module)

            export_args = export_args.parse_args(export_arg_list)
            return export_cmd.run(export_args)

        elif args.build_docs:
            console.print("📚 Building documentation from notebooks...")
            return self._run_nbdev_tool("nbdev-docs", "Docs", console)

        elif args.test:
            console.print("🧪 Running notebook tests...")
            return self._run_nbdev_tool("nbdev-test", "Test", console)

        elif args.clean:
            console.print("🧹 Cleaning notebook outputs...")
            return self._run_nbdev_tool("nbdev-clean", "Clean", console)

        else:
            console.print(Panel("[yellow]⚠️  No nbdev action specified. Use --export, --build-docs, --test, or --clean[/yellow]",
                              title="No Action", border_style="yellow"))
            return 1

    def _run_nbdev_tool(self, command: str, label: str, console) -> int:
        """Run one of nbdev's own CLI tools (nbdev-docs, nbdev-test, nbdev-clean).

        nbdev 3.x registers these console scripts with hyphens
        (nbdev-docs, not nbdev_docs); calling the underscore form raises
        FileNotFoundError since it has never existed under that name.
        """
        try:
            result = subprocess.run(
                [command], capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
        except FileNotFoundError:
            console.print(Panel(
                f"[red]❌ '{command}' not found[/red]\n\n"
                "This requires nbdev to be installed:\n"
                "  [cyan]pip install nbdev[/cyan]",
                title=f"{label} Error", border_style="red"
            ))
            return 1

        if result.returncode == 0:
            console.print(Panel(f"[green]✅ {label} succeeded![/green]",
                              title=f"{label} Success", border_style="green"))
        else:
            console.print(Panel(f"[red]❌ {label} failed: {result.stderr}[/red]",
                              title=f"{label} Error", border_style="red"))
        return result.returncode
