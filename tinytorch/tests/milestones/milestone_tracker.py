"""
Milestone Progress Tracker for TinyTorch

Tracks which modules students have completed and unlocks milestone tests accordingly.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

# Milestone definitions
MILESTONES = {
    "perceptron": {
        "name": "1957 - The Perceptron",
        "requires": ["00_setup", "01_tensor", "02_autograd"],
        "test": "test_perceptron_learning",
        "description": "First learning algorithm with automatic weight updates",
        "unlock_message": "🎉 You can now verify that gradient descent actually works!",
    },
    "xor": {
        "name": "1986 - Backpropagation (XOR)",
        "requires": ["00_setup", "01_tensor", "02_autograd", "03_nn"],
        "test": "test_xor_learning",
        "description": "Solving non-linear problems with hidden layers",
        "unlock_message": "🎉 You can now solve the problem that stumped neural networks for decades!",
    },
    "mlp_digits": {
        "name": "1989 - MLP on Real Data",
        "requires": ["00_setup", "01_tensor", "02_autograd", "03_nn", "04_training"],
        "test": "test_mlp_digits_learning",
        "description": "Scaling to real-world image classification",
        "unlock_message": "🎉 You can now train networks on real datasets!",
    },
    "cnn": {
        "name": "1998 - Convolutional Networks",
        "requires": ["00_setup", "01_tensor", "02_autograd", "03_nn", "04_training", "07_spatial"],
        "test": "test_cnn_learning",
        "description": "Preserving spatial structure in images",
        "unlock_message": "🎉 You can now build networks that understand spatial relationships!",
    },
    "transformer": {
        "name": "2017 - Transformer (Attention)",
        "requires": ["00_setup", "01_tensor", "02_autograd", "03_nn", "04_training", "11_embeddings", "12_attention"],
        "test": "test_transformer_learning",
        "description": "Attention mechanism for sequence processing",
        "unlock_message": "🎉 You can now build the architecture that powers modern language models!",
    },
}

MILESTONE_ORDER = ["perceptron", "xor", "mlp_digits", "cnn", "transformer"]


class MilestoneTracker:
    """Tracks student progress through TinyTorch modules and milestones."""

    def __init__(self, progress_file: Optional[Path] = None):
        if progress_file is None:
            progress_file = Path.home() / ".tinytorch" / "progress.json"

        self.progress_file = progress_file
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.progress = self._load_progress()

    def _load_progress(self) -> Dict:
        """Load progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"completed_modules": [], "unlocked_milestones": [], "completed_milestones": []}

    def _save_progress(self):
        """Save progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def mark_module_complete(self, module_name: str):
        """Mark a module as complete and check for unlocked milestones."""
        if module_name not in self.progress["completed_modules"]:
            self.progress["completed_modules"].append(module_name)
            self._save_progress()

            # Check for newly unlocked milestones
            newly_unlocked = self._check_unlocked_milestones()

            if newly_unlocked:
                for milestone_id in newly_unlocked:
                    self._show_unlock_message(milestone_id)

    def _check_unlocked_milestones(self) -> List[str]:
        """Check which milestones are newly unlocked."""
        newly_unlocked = []
        completed = set(self.progress["completed_modules"])

        for milestone_id in MILESTONE_ORDER:
            if milestone_id in self.progress["unlocked_milestones"]:
                continue

            milestone = MILESTONES[milestone_id]
            required = set(milestone["requires"])

            if required.issubset(completed):
                self.progress["unlocked_milestones"].append(milestone_id)
                newly_unlocked.append(milestone_id)

        if newly_unlocked:
            self._save_progress()

        return newly_unlocked

    def _show_unlock_message(self, milestone_id: str):
        """Show an exciting unlock message."""
        milestone = MILESTONES[milestone_id]

        console.print()
        console.print(Panel.fit(
            f"[bold green]🔓 MILESTONE UNLOCKED![/bold green]\n\n"
            f"[bold cyan]{milestone['name']}[/bold cyan]\n"
            f"{milestone['description']}\n\n"
            f"{milestone['unlock_message']}\n\n"
            f"[bold]Run the verification test:[/bold]\n"
            f"[yellow]pytest tests/milestones/test_learning_verification.py::{milestone['test']} -v[/yellow]",
            border_style="green",
            box=box.DOUBLE
        ))
        console.print()

    def show_progress(self):
        """Display current progress."""
        table = Table(title="🎯 TinyTorch Milestone Progress", box=box.ROUNDED)
        table.add_column("Milestone", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Requirements", style="dim")

        for milestone_id in MILESTONE_ORDER:
            milestone = MILESTONES[milestone_id]

            if milestone_id in self.progress["completed_milestones"]:
                status = "[green]✅ Completed[/green]"
            elif milestone_id in self.progress["unlocked_milestones"]:
                status = "[yellow]🔓 Unlocked[/yellow]"
            else:
                status = "[dim]🔒 Locked[/dim]"

            # Check which requirements are met
            completed = set(self.progress["completed_modules"])
            required = milestone["requires"]
            met = sum(1 for req in required if req in completed)
            req_str = f"{met}/{len(required)} modules"

            table.add_row(milestone["name"], status, req_str)

        console.print(table)
        console.print()

        # Show next steps
        next_milestone = self._get_next_milestone()
        if next_milestone:
            milestone_id, milestone = next_milestone
            console.print(Panel(
                f"[bold]Next Milestone:[/bold] {milestone['name']}\n\n"
                f"[bold]Complete these modules:[/bold]\n" +
                "\n".join(f"  • {req}" for req in milestone["requires"]
                         if req not in self.progress["completed_modules"]),
                title="📍 What's Next?",
                border_style="blue"
            ))

    def _get_next_milestone(self) -> Optional[tuple]:
        """Get the next milestone to unlock."""
        for milestone_id in MILESTONE_ORDER:
            if milestone_id not in self.progress["unlocked_milestones"]:
                return (milestone_id, MILESTONES[milestone_id])
        return None

    def mark_milestone_complete(self, milestone_id: str):
        """Mark a milestone test as passed."""
        if milestone_id not in self.progress["completed_milestones"]:
            self.progress["completed_milestones"].append(milestone_id)
            self._save_progress()

            milestone = MILESTONES[milestone_id]
            console.print()
            console.print(Panel.fit(
                f"[bold green]🏆 MILESTONE COMPLETED![/bold green]\n\n"
                f"[bold cyan]{milestone['name']}[/bold cyan]\n\n"
                f"You've successfully verified that your implementation works!\n"
                f"Your neural network actually learns. 🎓",
                border_style="green",
                box=box.DOUBLE
            ))
            console.print()

    def can_run_milestone(self, milestone_id: str) -> bool:
        """Check if a milestone test can be run."""
        return milestone_id in self.progress["unlocked_milestones"]

    def list_unlocked_tests(self):
        """Show all unlocked milestone tests."""
        if not self.progress["unlocked_milestones"]:
            console.print("[yellow]No milestones unlocked yet. Complete more modules![/yellow]")
            return

        console.print("[bold]🔓 Unlocked Milestone Tests:[/bold]\n")
        for milestone_id in self.progress["unlocked_milestones"]:
            if milestone_id in self.progress["completed_milestones"]:
                continue

            milestone = MILESTONES[milestone_id]
            console.print(f"[cyan]• {milestone['name']}[/cyan]")
            console.print(f"  [dim]{milestone['description']}[/dim]")
            console.print(f"  [yellow]pytest tests/milestones/test_learning_verification.py::{milestone['test']} -v[/yellow]\n")


def check_module_export(module_name: str, console=None):
    """
    Called after a student exports a module.
    Checks if this unlocks any milestones.

    Returns:
        dict: {
            'newly_unlocked': [milestone_ids],
            'messages': [unlock messages to display]
        }
    """
    tracker = MilestoneTracker()

    # Mark module complete and get newly unlocked milestones
    if module_name not in tracker.progress["completed_modules"]:
        tracker.progress["completed_modules"].append(module_name)
        tracker._save_progress()

    # Check for newly unlocked milestones
    newly_unlocked = tracker._check_unlocked_milestones()

    result = {
        'newly_unlocked': newly_unlocked,
        'messages': []
    }

    # Generate messages for each newly unlocked milestone
    for milestone_id in newly_unlocked:
        milestone = MILESTONES[milestone_id]
        message = (
            f"🔓 MILESTONE UNLOCKED!\n\n"
            f"{milestone['name']}\n"
            f"{milestone['description']}\n\n"
            f"{milestone['unlock_message']}\n\n"
            f"Run: tito milestones run {milestone_id}"
        )
        result['messages'].append(message)

        # Show message if console provided
        if console:
            from rich.panel import Panel
            from rich import box
            console.print()
            console.print(Panel.fit(
                f"[bold green]🔓 MILESTONE UNLOCKED![/bold green]\n\n"
                f"[bold cyan]{milestone['name']}[/bold cyan]\n"
                f"{milestone['description']}\n\n"
                f"{milestone['unlock_message']}\n\n"
                f"[bold]Run the verification test:[/bold]\n"
                f"[yellow]tito milestones run {milestone_id}[/yellow]",
                border_style="green",
                box=box.DOUBLE
            ))
            console.print()

    return result


def show_progress():
    """Show current milestone progress."""
    tracker = MilestoneTracker()
    tracker.show_progress()


def list_tests():
    """List all unlocked milestone tests."""
    tracker = MilestoneTracker()
    tracker.list_unlocked_tests()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "progress":
            show_progress()
        elif command == "list":
            list_tests()
        elif command == "complete":
            if len(sys.argv) > 2:
                tracker = MilestoneTracker()
                tracker.mark_module_complete(sys.argv[2])
        else:
            console.print(f"[red]Unknown command: {command}[/red]")
    else:
        show_progress()
