"""
Compatibility milestone progress tracker for TinyTorch tests and older hooks.

The canonical milestone definitions live in ``tito.commands.milestone``.
This file mirrors that table so legacy imports do not carry stale module
names or write to a separate home-directory progress file.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tito.commands.milestone import MILESTONE_SCRIPTS

console = Console()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _module_number(module_name: str) -> Optional[str]:
    prefix = str(module_name).split("_", 1)[0]
    try:
        return f"{int(prefix):02d}"
    except ValueError:
        return None


def _build_milestones() -> Dict[str, Dict]:
    milestones = {}
    for milestone_id, milestone in sorted(MILESTONE_SCRIPTS.items()):
        milestones[milestone_id] = {
            "name": f"{milestone['year']} - {milestone['name']}",
            "requires": [f"{m:02d}" for m in milestone["required_modules"]],
            "description": milestone["description"],
            "run_command": f"tito milestone run {milestone_id}",
        }
    return milestones


MILESTONES = _build_milestones()
MILESTONE_ORDER = sorted(MILESTONES.keys())


class MilestoneTracker:
    """Tracks module progress and milestone readiness using .tito files."""

    def __init__(
        self,
        progress_file: Optional[Path] = None,
        module_progress_file: Optional[Path] = None,
    ):
        root = _project_root()
        self.progress_file = progress_file or root / ".tito" / "milestones.json"
        self.module_progress_file = module_progress_file or root / ".tito" / "progress.json"
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.module_progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.progress = self._load_progress()

    def _load_progress(self) -> Dict:
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    progress = json.load(f)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                progress = {}
        else:
            progress = {}

        progress.setdefault("completed_milestones", [])
        progress.setdefault("unlocked_milestones", [])
        progress.setdefault("completion_dates", {})
        progress.setdefault("unlock_dates", {})
        progress.setdefault("achievements", [])
        return progress

    def _save_progress(self) -> None:
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.progress, f, indent=2)

    def _load_completed_modules(self) -> List[str]:
        if not self.module_progress_file.exists():
            return []
        try:
            with open(self.module_progress_file, "r", encoding="utf-8") as f:
                progress = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return []

        completed = []
        for module_name in progress.get("completed_modules", []):
            module_num = _module_number(module_name)
            if module_num is not None:
                completed.append(module_num)
        return completed

    def _save_completed_modules(self, completed_modules: List[str]) -> None:
        progress = {}
        if self.module_progress_file.exists():
            try:
                with open(self.module_progress_file, "r", encoding="utf-8") as f:
                    progress = json.load(f)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                progress = {}
        progress["completed_modules"] = sorted(set(completed_modules))
        with open(self.module_progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

    def mark_module_complete(self, module_name: str) -> List[str]:
        """Mark a module complete and return newly runnable milestone IDs."""
        module_num = _module_number(module_name)
        if module_num is None:
            return []

        completed = self._load_completed_modules()
        if module_num not in completed:
            completed.append(module_num)
            self._save_completed_modules(completed)

        newly_unlocked = self._check_unlocked_milestones()
        for milestone_id in newly_unlocked:
            self._show_unlock_message(milestone_id)
        return newly_unlocked

    def _check_unlocked_milestones(self) -> List[str]:
        newly_unlocked = []
        completed = set(self._load_completed_modules())
        unlocked = set(self.progress["unlocked_milestones"])
        completed_milestones = set(self.progress["completed_milestones"])

        for milestone_id in MILESTONE_ORDER:
            if milestone_id in unlocked or milestone_id in completed_milestones:
                continue
            required = set(MILESTONES[milestone_id]["requires"])
            if required.issubset(completed):
                unlocked.add(milestone_id)
                newly_unlocked.append(milestone_id)

        if newly_unlocked:
            self.progress["unlocked_milestones"] = sorted(unlocked)
            self.progress["total_unlocked"] = len(unlocked)
            self._save_progress()

        return newly_unlocked

    def _show_unlock_message(self, milestone_id: str) -> None:
        milestone = MILESTONES[milestone_id]
        console.print()
        console.print(Panel.fit(
            f"[bold green]Milestone ready to run[/bold green]\n\n"
            f"[bold cyan]{milestone['name']}[/bold cyan]\n"
            f"{milestone['description']}\n\n"
            f"[bold]Run:[/bold] [yellow]{milestone['run_command']}[/yellow]",
            border_style="green",
            box=box.DOUBLE,
        ))
        console.print()

    def show_progress(self) -> None:
        table = Table(title="TinyTorch Milestone Progress", box=box.ROUNDED)
        table.add_column("Milestone", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Requirements", style="dim")

        completed = set(self._load_completed_modules())
        for milestone_id in MILESTONE_ORDER:
            milestone = MILESTONES[milestone_id]
            if milestone_id in self.progress["completed_milestones"]:
                status = "[green]Completed[/green]"
            elif milestone_id in self.progress["unlocked_milestones"]:
                status = "[yellow]Ready[/yellow]"
            else:
                status = "[dim]Locked[/dim]"

            required = milestone["requires"]
            met = sum(1 for req in required if req in completed)
            table.add_row(milestone["name"], status, f"{met}/{len(required)} modules")

        console.print(table)

    def mark_milestone_complete(self, milestone_id: str) -> None:
        if milestone_id not in MILESTONES:
            raise ValueError(f"Unknown milestone: {milestone_id}")
        if milestone_id not in self.progress["completed_milestones"]:
            self.progress["completed_milestones"].append(milestone_id)
        if milestone_id not in self.progress["unlocked_milestones"]:
            self.progress["unlocked_milestones"].append(milestone_id)
        self.progress["completed_milestones"] = sorted(set(self.progress["completed_milestones"]))
        self.progress["unlocked_milestones"] = sorted(set(self.progress["unlocked_milestones"]))
        self.progress["total_unlocked"] = len(self.progress["unlocked_milestones"])
        self._save_progress()

    def can_run_milestone(self, milestone_id: str) -> bool:
        return milestone_id in self.progress["unlocked_milestones"]

    def list_unlocked_tests(self) -> None:
        unlocked = [
            mid for mid in self.progress["unlocked_milestones"]
            if mid not in self.progress["completed_milestones"]
        ]
        if not unlocked:
            console.print("[yellow]No milestones ready yet. Complete more modules.[/yellow]")
            return
        for milestone_id in unlocked:
            milestone = MILESTONES[milestone_id]
            console.print(f"[cyan]Milestone {milestone_id}: {milestone['name']}[/cyan]")
            console.print(f"  [yellow]{milestone['run_command']}[/yellow]\n")


def check_module_export(module_name: str, console=None):
    """Legacy hook called after a student exports a module."""
    tracker = MilestoneTracker()
    newly_unlocked = tracker.mark_module_complete(module_name)

    result = {"newly_unlocked": newly_unlocked, "messages": []}
    for milestone_id in newly_unlocked:
        milestone = MILESTONES[milestone_id]
        message = (
            "Milestone ready to run\n\n"
            f"{milestone['name']}\n"
            f"{milestone['description']}\n\n"
            f"Run: {milestone['run_command']}"
        )
        result["messages"].append(message)
        if console:
            console.print()
            console.print(Panel.fit(
                f"[bold green]Milestone ready to run[/bold green]\n\n"
                f"[bold cyan]{milestone['name']}[/bold cyan]\n"
                f"{milestone['description']}\n\n"
                f"[bold]Run:[/bold] [yellow]{milestone['run_command']}[/yellow]",
                border_style="green",
                box=box.DOUBLE,
            ))
            console.print()
    return result


def show_progress():
    MilestoneTracker().show_progress()


def list_tests():
    MilestoneTracker().list_unlocked_tests()


if __name__ == "__main__":
    show_progress()
