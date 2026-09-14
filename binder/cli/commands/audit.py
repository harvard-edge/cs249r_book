"""Chapter-level build audits (`binder audit chapter-pdf|chapter-html`)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def _repo_root() -> Path:
    """Return the repository root that contains ``binder/``, resolved from this file."""
    return Path(__file__).resolve().parents[3]


def _load_module(name: str, path: Path):
    """Import the Python file *path* as module *name* and register it in ``sys.modules``."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class AuditCommand:
    """Per-chapter PDF/HTML build audits with ledger tracking."""

    def __init__(self, config_manager, chapter_discovery):
        """Store shared managers and locate the ``binder/tools/audit`` script directory."""
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery
        self.repo_root = _repo_root()
        self.audit_dir = self.repo_root / "binder" / "tools" / "audit"

    def run(self, args: list[str]) -> bool:
        """Dispatch ``chapter-pdf`` or ``chapter-html`` to its audit script.

        No arguments or a help flag prints help and returns True. An unknown
        target prints an error plus help and returns False.
        """
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
        """Load an audit script and call its ``main()`` with *args* as ``sys.argv``.

        ``sys.argv`` is restored afterward. Returns True only when ``main()``
        returns 0, and False when the script file does not exist.
        """
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
        """Print the audit targets and usage examples."""
        console.print("[bold cyan]binder audit[/bold cyan] — per-chapter build audits\n")
        console.print("  [green]chapter-pdf[/green]  Build + audit one chapter PDF (ledger under artifacts/)")
        console.print("  [green]chapter-html[/green] Build + audit one chapter HTML (ledger under artifacts/)\n")
        console.print("[dim]Examples:[/dim]")
        console.print("  ./binder/binder audit chapter-pdf --vol1 training")
        console.print("  ./binder/binder audit chapter-pdf --vol1 --all")
        console.print("  ./binder/binder audit chapter-html --list")
        console.print("  ./binder/binder audit chapter-pdf --report")
