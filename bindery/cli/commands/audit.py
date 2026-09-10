"""Chapter-level build audits (`binder audit chapter-pdf|chapter-html`)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class AuditCommand:
    """Per-chapter PDF/HTML build audits with ledger tracking."""

    def __init__(self, config_manager, chapter_discovery):
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery
        self.repo_root = _repo_root()
        self.audit_dir = self.repo_root / "book" / "tools" / "audit"

    def run(self, args: list[str]) -> bool:
        if not args or args[0] in ("-h", "--help", "help"):
            self._print_help()
            return True

        target = args[0]
        rest = args[1:]
        if target == "chapter-pdf":
            return self._run_chapter_audit("chapter_pdf_verify", rest)
        if target == "chapter-html":
            return self._run_chapter_audit("chapter_html_verify", rest)
        console.print(f"[red]Unknown audit target: {target}[/red]")
        self._print_help()
        return False

    def _run_chapter_audit(self, module_name: str, args: list[str]) -> bool:
        script = self.audit_dir / f"{module_name}.py"
        if not script.exists():
            console.print(f"[red]Audit module not found: {script}[/red]")
            return False
        mod = _load_module(module_name, script)
        old_argv = sys.argv
        try:
            sys.argv = [str(script), *args]
            return mod.main() == 0
        finally:
            sys.argv = old_argv

    def _print_help(self) -> None:
        console.print("[bold cyan]binder audit[/bold cyan] — per-chapter build audits\n")
        console.print("  [green]chapter-pdf[/green]  Build + audit one chapter PDF (ledger under artifacts/)")
        console.print("  [green]chapter-html[/green] Build + audit one chapter HTML (ledger under artifacts/)\n")
        console.print("[dim]Examples:[/dim]")
        console.print("  ./binder audit chapter-pdf --vol1 training")
        console.print("  ./binder audit chapter-pdf --vol1 --all")
        console.print("  ./binder audit chapter-html --list")
        console.print("  ./binder audit chapter-pdf --report")
