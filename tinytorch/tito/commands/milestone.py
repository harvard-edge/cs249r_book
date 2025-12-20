"""
Milestone command group for TinyTorch CLI: capability-based learning progression.

The milestone system transforms module completion into meaningful capability achievements.
Instead of just finishing modules, students unlock epic milestones that represent
real-world ML engineering skills.
"""

from argparse import ArgumentParser, Namespace
from rich.panel import Panel
from rich import box
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, TimeElapsedColumn
from rich.console import Console
from rich.align import Align
from rich.text import Text
from rich.layout import Layout
from rich.tree import Tree
from rich.columns import Columns
import sys
import json
import time
import subprocess
import yaml
from datetime import datetime
from pathlib import Path

from .base import BaseCommand
from ..core.console import print_ascii_logo
from ..core.console import get_console


# Name aliases for milestone IDs (allows `tito milestone run perceptron`)
MILESTONE_ALIASES = {
    "perceptron": "01",
    "xor": "02",
    "mlp": "03",
    "cnn": "04",
    "transformer": "05",
    "mlperf": "06",
    "olympics": "06",
}

# Milestone-to-script mapping for tito milestone run command
MILESTONE_SCRIPTS = {
    "01": {
        "id": "01",
        "name": "Perceptron (1957)",
        "year": 1957,
        "title": "Frank Rosenblatt's First Neural Network",
        "script": "milestones/01_1957_perceptron/01_rosenblatt_forward.py",
        "required_modules": [1, 2, 3],  # Tensor, Activations, Layers (forward pass only)
        "description": "Build the first neural network (forward pass)",
        "historical_context": "Rosenblatt's perceptron proved machines could learn",
        "emoji": "🧠"
    },
    "02": {
        "id": "02",
        "name": "XOR Crisis (1969)",
        "year": 1969,
        "title": "The Problem That Stalled AI",
        "script": "milestones/02_1969_xor/01_xor_crisis.py",
        "required_modules": [1, 2, 3],  # Just forward pass: Tensor, Activations, Layers
        "description": "Single-layer perceptron CANNOT solve XOR (75% max)",
        "historical_context": "Minsky & Papert proved limits of single-layer networks",
        "emoji": "🔀"
    },
    "03": {
        "id": "03",
        "name": "MLP Revival (1986)",
        "year": 1986,
        "title": "Backpropagation Breakthrough",
        "scripts": [
            {
                "name": "XOR Solved",
                "script": "milestones/02_1969_xor/02_xor_solved.py",
                "description": "Hidden layers + backprop SOLVE the impossible XOR problem!",
                "required_modules": [1, 2, 3, 4, 5, 6, 7, 8]  # Full training: Tensor through Training
            },
            {
                "name": "TinyDigits",
                "script": "milestones/03_1986_mlp/01_rumelhart_tinydigits.py",
                "description": "Scale up to real data - handwritten digit recognition",
                "required_modules": [1, 2, 3, 4, 5, 6, 7, 8]  # Full training infrastructure
            }
        ],
        "required_modules": [1, 2, 3, 4, 5, 6, 7, 8],  # Full training for XOR Solved
        "description": "Solve XOR with hidden layers, then train on real data",
        "historical_context": "Rumelhart, Hinton & Williams (Nature, 1986) ended the AI Winter",
        "emoji": "🎓"
    },
    "04": {
        "id": "04",
        "name": "CNN Revolution (1998)",
        "year": 1998,
        "title": "LeNet - Computer Vision Breakthrough",
        "scripts": [
            {
                "name": "TinyDigits",
                "script": "milestones/04_1998_cnn/01_lecun_tinydigits.py",
                "description": "Prove CNNs > MLPs on synthetic 8x8 digits (works offline)",
                "required_modules": [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Full training + Convolutions
            },
            {
                "name": "CIFAR-10",
                "script": "milestones/04_1998_cnn/02_lecun_cifar10.py",
                "description": "Scale to natural images with YOUR DataLoader (requires download)",
                "required_modules": [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Full training + Convolutions
            }
        ],
        "required_modules": [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Full training + Convolutions
        "description": "Build LeNet for digit recognition, then scale to natural images",
        "historical_context": "Yann LeCun's convolutional networks revolutionized computer vision",
        "emoji": "👁️"
    },
    "05": {
        "id": "05",
        "name": "Transformer Era (2017)",
        "year": 2017,
        "title": "Attention is All You Need",
        "script": "milestones/05_2017_transformer/01_vaswani_attention.py",
        "required_modules": [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13],  # Full training + Embeddings, Attention, Transformers
        "description": "Prove attention works with sequence reversal",
        "historical_context": "Vaswani et al. revolutionized NLP",
        "emoji": "🤖"
    },
    "06": {
        "id": "06",
        "name": "MLPerf Benchmarks (2018)",
        "year": 2018,
        "title": "The Optimization Olympics",
        "scripts": [
            {
                "name": "Model Compression",
                "script": "milestones/06_2018_mlperf/01_optimization_olympics.py",
                "description": "Profiling + Quantization + Pruning on MLP",
                "required_modules": [1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19]  # Full training + Optimization tier
            },
            {
                "name": "Generation Speedup",
                "script": "milestones/06_2018_mlperf/02_generation_speedup.py",
                "description": "KV Caching for 10x faster Transformer",
                "required_modules": [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 18]  # Full training + Embeddings + Attention + Profiler + Memoization (18)
            }
        ],
        "required_modules": [1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19],  # Full training + Optimization tier
        "description": "Compress and accelerate your neural network",
        "historical_context": "MLPerf standardized ML benchmarks",
        "emoji": "🏆"
    }
}


class MilestoneSystem:
    """Core milestone tracking and management system."""

    def __init__(self, config):
        self.config = config
        self.console = get_console()

        # Load milestones from configuration file
        self.MILESTONES = self._load_milestones_config()

    def _load_milestones_config(self) -> dict:
        """Load milestone configuration from YAML files (main and era-specific)."""
        config_path = Path("milestones") / "milestones.yml"
        milestones = {}

        # Try to load main milestones.yml first
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # Convert to expected format
                for milestone_id, milestone_data in config['milestones'].items():
                    milestone_data['id'] = str(milestone_id)
                    milestones[str(milestone_id)] = milestone_data

            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not load main milestone config: {e}[/yellow]")

        # Also try to load era-specific configurations
        era_paths = [
            Path("milestones") / "foundation" / "milestone.yml",
            Path("milestones") / "revolution" / "milestone.yml",
            Path("milestones") / "generation" / "milestone.yml"
        ]

        for era_path in era_paths:
            if era_path.exists():
                try:
                    with open(era_path, 'r') as f:
                        era_config = yaml.safe_load(f)

                    if 'milestone' in era_config:
                        milestone_data = era_config['milestone']
                        milestone_id = milestone_data['id']
                        milestones[str(milestone_id)] = milestone_data

                except Exception as e:
                    self.console.print(f"[yellow]Warning: Could not load era config {era_path}: {e}[/yellow]")

        # If no milestones loaded, use MILESTONE_SCRIPTS as fallback
        if not milestones:
            return MILESTONE_SCRIPTS

        return milestones

    def get_milestone_status(self) -> dict:
        """Get current milestone progress status."""
        milestone_data = self._get_milestone_progress_data()

        status = {
            "milestones": {},
            "overall_progress": 0,
            "total_unlocked": 0,
            "next_milestone": None
        }

        total_milestones = len(self.MILESTONES)
        unlocked_count = 0

        for milestone_id, milestone in self.MILESTONES.items():
            # Check if all required modules are complete (no more checkpoint dependencies)
            required_modules = milestone.get("required_modules", [])
            required_complete = all(
                self._is_module_completed(f"{mod:02d}")
                for mod in required_modules
            )

            # Check if milestone is unlocked
            is_unlocked = milestone_id in milestone_data.get("unlocked_milestones", [])

            # Check if trigger module is completed (if trigger_module exists)
            trigger_module = milestone.get("trigger_module", "")
            if trigger_module:
                trigger_complete = self._is_module_completed(trigger_module)
            else:
                # No trigger module - consider complete if all required modules done
                trigger_complete = required_complete

            milestone_status = {
                "id": milestone_id,
                "name": milestone["name"],
                "title": milestone["title"],
                "emoji": milestone.get("emoji", "🎯"),
                "trigger_module": trigger_module,
                "required_modules": milestone.get("required_modules", []),
                "victory_condition": milestone.get("victory_condition", milestone.get("description", "")),
                "capability": milestone.get("capability", ""),
                "real_world_impact": milestone.get("real_world_impact", ""),
                "required_complete": required_complete,
                "trigger_complete": trigger_complete,
                "is_unlocked": is_unlocked,
                "can_unlock": required_complete and trigger_complete and not is_unlocked,
                "unlock_date": milestone_data.get("unlock_dates", {}).get(milestone_id)
            }

            status["milestones"][milestone_id] = milestone_status

            if is_unlocked:
                unlocked_count += 1
            elif milestone_status["can_unlock"] and not status["next_milestone"]:
                status["next_milestone"] = milestone_id

        status["total_unlocked"] = unlocked_count
        status["overall_progress"] = (unlocked_count / total_milestones) * 100 if total_milestones > 0 else 0

        return status

    def check_milestone_unlock(self, completed_module: str) -> dict:
        """Check if completing a module unlocks a milestone."""
        result = {
            "milestone_unlocked": False,
            "milestone_id": None,
            "milestone_data": None,
            "celebration_needed": False
        }

        # Find milestone triggered by this module
        for milestone_id, milestone in self.MILESTONES.items():
            if milestone["trigger_module"] == completed_module:
                status = self.get_milestone_status()
                milestone_status = status["milestones"][milestone_id]

                if milestone_status["can_unlock"]:
                    # Unlock the milestone!
                    self._unlock_milestone(milestone_id)
                    result.update({
                        "milestone_unlocked": True,
                        "milestone_id": milestone_id,
                        "milestone_data": milestone,
                        "celebration_needed": True
                    })
                break

        return result

    def run_milestone_test(self, milestone_id: str) -> dict:
        """Run tests to validate milestone achievement."""
        if milestone_id not in self.MILESTONES:
            return {"success": False, "error": f"Milestone {milestone_id} not found"}

        milestone = self.MILESTONES[milestone_id]

        # Check all required modules are complete
        required_modules = milestone.get("required_modules", [])
        failed_modules = []

        for mod in required_modules:
            if not self._is_module_completed(f"{mod:02d}"):
                failed_modules.append(f"{mod:02d}")

        if failed_modules:
            return {
                "success": False,
                "error": f"Required modules not completed: {', '.join(failed_modules)}",
                "milestone_name": milestone["name"]
            }

        # Check trigger module completion
        trigger_module = milestone.get("trigger_module", "")
        if trigger_module and not self._is_module_completed(trigger_module):
            return {
                "success": False,
                "error": f"Trigger module {trigger_module} not completed",
                "milestone_name": milestone["name"]
            }

        # All tests passed
        return {
            "success": True,
            "milestone_id": milestone_id,
            "milestone_name": milestone["name"],
            "title": milestone.get("title", ""),
            "capability": milestone.get("capability", ""),
            "victory_condition": milestone.get("victory_condition", "")
        }

    def _unlock_milestone(self, milestone_id: str) -> None:
        """Record milestone unlock in progress tracking."""
        milestone_data = self._get_milestone_progress_data()

        if milestone_id not in milestone_data["unlocked_milestones"]:
            milestone_data["unlocked_milestones"].append(milestone_id)
            milestone_data["unlock_dates"][milestone_id] = datetime.now().isoformat()
            milestone_data["total_unlocked"] = len(milestone_data["unlocked_milestones"])

        self._save_milestone_progress_data(milestone_data)

    def _is_module_completed(self, module_name: str) -> bool:
        """Check if a module has been completed."""
        # Check module progress file
        progress_file = Path(".tito") / "progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    return module_name in progress_data.get("completed_modules", [])
            except (json.JSONDecodeError, IOError):
                pass
        return False

    def _get_milestone_progress_data(self) -> dict:
        """Get or create milestone progress data."""
        progress_dir = Path(".tito")
        progress_file = progress_dir / "milestones.json"

        progress_dir.mkdir(exist_ok=True)

        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return {
            "unlocked_milestones": [],
            "unlock_dates": {},
            "total_unlocked": 0,
            "achievements": []
        }

    def _save_milestone_progress_data(self, milestone_data: dict) -> None:
        """Save milestone progress data."""
        progress_dir = Path(".tito")
        progress_file = progress_dir / "milestones.json"

        progress_dir.mkdir(exist_ok=True)

        try:
            with open(progress_file, 'w') as f:
                json.dump(milestone_data, f, indent=2)
        except IOError:
            pass


class MilestoneCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "milestone"

    @property
    def description(self) -> str:
        return "Milestone achievement and capability unlock commands"

    def add_arguments(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest='milestone_command',
            help='Milestone subcommands',
            metavar='SUBCOMMAND'
        )

        # List subcommand (NEW)
        list_parser = subparsers.add_parser(
            'list',
            help='List available milestones and their status'
        )
        list_parser.add_argument(
            '--simple',
            action='store_true',
            help='Show simple list (less detail)'
        )

        # Run subcommand (NEW)
        run_parser = subparsers.add_parser(
            'run',
            help='Run a milestone with prerequisite checking'
        )
        run_parser.add_argument(
            'milestone_id',
            help='Milestone ID (01-06) or name (perceptron, xor, mlp, cnn, transformer, mlperf)'
        )
        run_parser.add_argument(
            '--part',
            type=int,
            help='Run only a specific part (for multi-part milestones)'
        )
        run_parser.add_argument(
            '--skip-checks',
            action='store_true',
            help='Skip prerequisite checks (not recommended)'
        )

        # Info subcommand (NEW)
        info_parser = subparsers.add_parser(
            'info',
            help='Show detailed information about a milestone'
        )
        info_parser.add_argument(
            'milestone_id',
            help='Milestone ID (01-06) or name (perceptron, xor, mlp, cnn, transformer, mlperf)'
        )

        # Status subcommand
        status_parser = subparsers.add_parser(
            'status',
            help='View milestone progress and achievements'
        )
        status_parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed milestone information'
        )

        # Timeline subcommand
        timeline_parser = subparsers.add_parser(
            'timeline',
            help='View milestone timeline and progression'
        )
        timeline_parser.add_argument(
            '--horizontal',
            action='store_true',
            help='Show horizontal progress bar instead of tree'
        )

        # Test subcommand
        test_parser = subparsers.add_parser(
            'test',
            help='Test milestone achievement requirements'
        )
        test_parser.add_argument(
            'milestone_id',
            nargs='?',
            help='Milestone ID to test (1-5), or test next available'
        )

        # Demo subcommand
        demo_parser = subparsers.add_parser(
            'demo',
            help='Run milestone capability demonstration'
        )
        demo_parser.add_argument(
            'milestone_id',
            help='Milestone ID to demonstrate (1-5)'
        )

    def run(self, args: Namespace) -> int:
        console = self.console

        if not hasattr(args, 'milestone_command') or not args.milestone_command:
            console.print(Panel(
                "[bold cyan]Milestone Commands[/bold cyan]\n\n"
                "Recreate ML history and achieve epic capabilities!\n\n"
                "Available subcommands:\n"
                "  • [bold]list[/bold]       - List available milestones\n"
                "  • [bold]run[/bold]        - Run a milestone (with prereq checks)\n"
                "  • [bold]info[/bold]       - Show detailed milestone information\n"
                "  • [bold]status[/bold]     - View progress and achievements\n"
                "  • [bold]timeline[/bold]   - View milestone timeline\n"
                "  • [bold]test[/bold]       - Test milestone requirements\n"
                "  • [bold]demo[/bold]       - Run capability demonstration\n\n"
                "[dim]Examples:[/dim]\n"
                "[dim]  tito milestone list[/dim]\n"
                "[dim]  tito milestone run 02           # Run all parts[/dim]\n"
                "[dim]  tito milestone run 02 --part 1  # Run Part 1 only[/dim]\n"
                "[dim]  tito milestone run 02 --part 2  # Run Part 2 only[/dim]\n"
                "[dim]  tito milestone info 03[/dim]\n"
                "[dim]  tito milestone status --detailed[/dim]",
                title="🏆 Milestone System",
                border_style="bright_cyan"
            ))
            return 0

        # Execute the appropriate subcommand
        if args.milestone_command == 'list':
            return self._handle_list_command(args)
        elif args.milestone_command == 'run':
            return self._handle_run_command(args)
        elif args.milestone_command == 'info':
            return self._handle_info_command(args)
        elif args.milestone_command == 'status':
            return self._handle_status_command(args)
        elif args.milestone_command == 'timeline':
            return self._handle_timeline_command(args)
        elif args.milestone_command == 'test':
            return self._handle_test_command(args)
        elif args.milestone_command == 'demo':
            return self._handle_demo_command(args)
        else:
            console.print(Panel(
                f"[red]Unknown milestone subcommand: {args.milestone_command}[/red]",
                title="Error",
                border_style="red"
            ))
            return 1

    def _handle_status_command(self, args: Namespace) -> int:
        """Handle milestone status command."""
        console = self.console
        milestone_system = MilestoneSystem(self.config)
        status = milestone_system.get_milestone_status()

        # Show header with overall progress
        total_milestones = len(milestone_system.MILESTONES)
        console.print(Panel(
            f"[bold cyan]🎮 TinyTorch Milestone Progress[/bold cyan]\n\n"
            f"[bold]Capabilities Unlocked:[/bold] {status['total_unlocked']}/{total_milestones} milestones\n"
            f"[bold]Overall Progress:[/bold] {status['overall_progress']:.0f}%\n\n"
            f"[dim]Transform from student to ML Systems Engineer![/dim]",
            title="🚀 Your Epic Journey",
            border_style="bright_blue"
        ))

        # Show milestone status
        for milestone_id in sorted(milestone_system.MILESTONES.keys()):
            milestone = status["milestones"][milestone_id]
            self._show_milestone_status(milestone, args.detailed)

        # Show next steps
        if status["next_milestone"]:
            next_milestone = status["milestones"][status["next_milestone"]]
            console.print(Panel(
                f"[bold cyan]🎯 Next Achievement[/bold cyan]\n\n"
                f"[bold yellow]{next_milestone['emoji']} {next_milestone['title']}[/bold yellow]\n"
                f"[dim]{next_milestone['victory_condition']}[/dim]\n\n"
                f"[green]Ready to unlock![/green] Complete: {next_milestone['trigger_module']}\n"
                f"[dim]tito module complete {next_milestone['trigger_module']}[/dim]",
                title="Next Milestone",
                border_style="bright_green"
            ))
        elif status["total_unlocked"] == 5:
            console.print(Panel(
                f"[bold green]🏆 QUEST COMPLETE! 🏆[/bold green]\n\n"
                f"[green]You've unlocked all 5 epic milestones![/green]\n"
                f"[bold white]You are now an ML Systems Engineer![/bold white]\n\n"
                f"[cyan]Share your achievement and inspire others![/cyan]",
                title="🌟 FULL MASTERY ACHIEVED",
                border_style="bright_green"
            ))

        return 0

    def _show_milestone_status(self, milestone: dict, detailed: bool = False) -> None:
        """Show status for a single milestone."""
        console = self.console

        # Status indicator
        if milestone["is_unlocked"]:
            status_icon = "🔓"
            status_color = "green"
            status_text = "UNLOCKED"
        elif milestone["can_unlock"]:
            status_icon = "⚡"
            status_color = "yellow"
            status_text = "READY TO UNLOCK"
        elif milestone["required_complete"] and not milestone["trigger_complete"]:
            status_icon = "🔒"
            status_color = "cyan"
            status_text = f"COMPLETE: {milestone['trigger_module']}"
        else:
            status_icon = "🔒"
            status_color = "dim"
            status_text = "LOCKED"

        # Basic display
        milestone_content = (
            f"[{status_color}]{status_icon} {milestone['emoji']} {milestone['title']}[/{status_color}]\n"
            f"[dim]{milestone['victory_condition']}[/dim]"
        )

        # Add detailed information if requested
        if detailed:
            req_status = "✅" if milestone["required_complete"] else "❌"
            trigger_status = "✅" if milestone["trigger_complete"] else "❌"

            required_modules_str = ', '.join(f"{m:02d}" for m in milestone.get('required_modules', []))

            milestone_content += (
                f"\n\n[bold]Requirements:[/bold]\n"
                f"  {req_status} Modules: {required_modules_str}\n"
                f"  {trigger_status} Trigger: {milestone.get('trigger_module', 'N/A')}\n"
                f"[bold]Capability:[/bold] {milestone['capability']}\n"
                f"[bold]Impact:[/bold] {milestone['real_world_impact']}"
            )

            if milestone["is_unlocked"] and milestone.get("unlock_date"):
                unlock_date = datetime.fromisoformat(milestone["unlock_date"]).strftime("%Y-%m-%d")
                milestone_content += f"\n[dim]Unlocked: {unlock_date}[/dim]"

        console.print(Panel(
            milestone_content,
            title=f"Milestone {milestone['id']}",
            border_style=status_color
        ))

    def _handle_timeline_command(self, args: Namespace) -> int:
        """Handle milestone timeline command."""
        console = self.console
        milestone_system = MilestoneSystem(self.config)
        status = milestone_system.get_milestone_status()

        if args.horizontal:
            self._show_horizontal_timeline(status, milestone_system)
        else:
            self._show_tree_timeline(status, milestone_system)

        return 0

    def _show_horizontal_timeline(self, status: dict, milestone_system: MilestoneSystem) -> None:
        """Show horizontal progress bar timeline."""
        console = self.console

        total_milestones = len(milestone_system.MILESTONES)
        console.print(Panel(
            f"[bold cyan]🎮 Milestone Timeline[/bold cyan]\n\n"
            f"[bold]Progress:[/bold] {status['total_unlocked']}/{total_milestones} milestones unlocked",
            title="Your Epic Journey",
            border_style="bright_blue"
        ))

        # Create progress bar
        progress_width = 50
        total_milestones = len(milestone_system.MILESTONES)
        unlocked_width = int((status["total_unlocked"] / total_milestones) * progress_width)

        # Create milestone markers
        timeline = []
        for milestone_id in sorted(milestone_system.MILESTONES.keys()):
            milestone = status["milestones"][milestone_id]

            if milestone["is_unlocked"]:
                marker = f"[green]{milestone['emoji']}[/green]"
            elif milestone["can_unlock"]:
                marker = f"[yellow blink]{milestone['emoji']}[/yellow blink]"
            else:
                marker = f"[dim]{milestone['emoji']}[/dim]"

            timeline.append(marker)

        # Show timeline
        console.print(f"\n{'  '.join(timeline)}")

        # Progress bar
        filled = "█" * unlocked_width
        empty = "░" * (progress_width - unlocked_width)
        console.print(f"\n[green]{filled}[/green][dim]{empty}[/dim]")
        console.print(f"[dim]{status['overall_progress']:.0f}% complete[/dim]\n")

    def _show_tree_timeline(self, status: dict, milestone_system: MilestoneSystem) -> None:
        """Show tree-style milestone timeline."""
        console = self.console

        console.print(Panel(
            f"[bold cyan]🎮 Milestone Progression Tree[/bold cyan]\n\n"
            f"[bold]Your journey from student to ML Systems Engineer[/bold]",
            title="Epic Timeline",
            border_style="bright_blue"
        ))

        # Create tree structure
        tree = Tree("🚀 [bold]TinyTorch Mastery Journey[/bold]")

        for milestone_id in sorted(milestone_system.MILESTONES.keys()):
            milestone = status["milestones"][milestone_id]

            if milestone["is_unlocked"]:
                node_style = "green"
                icon = "✅"
            elif milestone["can_unlock"]:
                node_style = "yellow"
                icon = "⚡"
            else:
                node_style = "dim"
                icon = "🔒"

            branch = tree.add(
                f"[{node_style}]{icon} {milestone['emoji']} {milestone['title']}[/{node_style}]"
            )

            # Add capability description
            branch.add(f"[dim]{milestone['capability']}[/dim]")

            # Add trigger module info
            if milestone["trigger_complete"]:
                branch.add(f"[green]✅ {milestone['trigger_module']} completed[/green]")
            else:
                branch.add(f"[dim]🎯 Complete: {milestone['trigger_module']}[/dim]")

        console.print(tree)
        console.print()

    def _handle_test_command(self, args: Namespace) -> int:
        """Handle milestone test command."""
        console = self.console
        milestone_system = MilestoneSystem(self.config)

        # Determine which milestone to test
        if args.milestone_id:
            milestone_id = args.milestone_id
        else:
            # Test next available milestone
            status = milestone_system.get_milestone_status()
            if status["next_milestone"]:
                milestone_id = status["next_milestone"]
            else:
                console.print(Panel(
                    "[yellow]No milestone available to test.[/yellow]\n\n"
                    "Either all milestones are unlocked or none are ready.\n"
                    "Use [dim]tito milestone status[/dim] to see your progress.",
                    title="No Test Available",
                    border_style="yellow"
                ))
                return 0

        # Validate milestone ID
        if milestone_id not in milestone_system.MILESTONES:
            console.print(Panel(
                f"[red]Invalid milestone ID: {milestone_id}[/red]\n\n"
                f"Valid milestone IDs: 1, 2, 3, 4, 5",
                title="Invalid Milestone",
                border_style="red"
            ))
            return 1

        milestone = milestone_system.MILESTONES[milestone_id]

        console.print(Panel(
            f"[bold cyan]🧪 Testing Milestone {milestone_id}[/bold cyan]\n\n"
            f"[bold]{milestone['emoji']} {milestone['title']}[/bold]\n"
            f"[dim]{milestone['victory_condition']}[/dim]",
            title="Milestone Test",
            border_style="bright_cyan"
        ))

        # Run the test with progress animation
        with console.status(f"[bold green]Testing milestone requirements...", spinner="dots"):
            result = milestone_system.run_milestone_test(milestone_id)

        # Show results
        if result["success"]:
            console.print(Panel(
                f"[bold green]✅ Milestone Test Passed![/bold green]\n\n"
                f"[green]All requirements met for {result['milestone_name']}[/green]\n"
                f"[cyan]Capability: {result['capability']}[/cyan]\n\n"
                f"[bold yellow]Complete the trigger module to unlock:[/bold yellow]\n"
                f"[dim]tito module complete {milestone['trigger_module']}[/dim]",
                title="🎉 Ready to Unlock!",
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"[bold yellow]⚠️ Milestone Requirements Not Met[/bold yellow]\n\n"
                f"[yellow]Milestone: {result.get('milestone_name', 'Unknown')}[/yellow]\n"
                f"[red]Issue: {result.get('error', 'Unknown error')}[/red]\n\n"
                f"[cyan]Complete the required modules and try again.[/cyan]",
                title="Requirements Missing",
                border_style="yellow"
            ))

        return 0

    def _handle_demo_command(self, args: Namespace) -> int:
        """Handle milestone demo command."""
        console = self.console
        milestone_system = MilestoneSystem(self.config)
        milestone_id = args.milestone_id

        # Validate milestone ID
        if milestone_id not in milestone_system.MILESTONES:
            console.print(Panel(
                f"[red]Invalid milestone ID: {milestone_id}[/red]\n\n"
                f"Valid milestone IDs: 1, 2, 3, 4, 5",
                title="Invalid Milestone",
                border_style="red"
            ))
            return 1

        milestone = milestone_system.MILESTONES[milestone_id]
        status = milestone_system.get_milestone_status()
        milestone_status = status["milestones"][milestone_id]

        # Check if milestone is unlocked
        if not milestone_status["is_unlocked"]:
            console.print(Panel(
                f"[yellow]Milestone {milestone_id} not yet unlocked.[/yellow]\n\n"
                f"[bold]{milestone['emoji']} {milestone['title']}[/bold]\n"
                f"[dim]{milestone['victory_condition']}[/dim]\n\n"
                f"[cyan]Complete the requirements first:[/cyan]\n"
                f"[dim]tito milestone test {milestone_id}[/dim]",
                title="Milestone Locked",
                border_style="yellow"
            ))
            return 0

        # Check if demo file exists
        demo_path = Path("capabilities") / milestone["demo_file"]
        if not demo_path.exists():
            console.print(Panel(
                f"[yellow]Demo not available for Milestone {milestone_id}[/yellow]\n\n"
                f"Demo file not found: {milestone['demo_file']}\n"
                f"[dim]This demo may be coming in a future update.[/dim]",
                title="Demo Unavailable",
                border_style="yellow"
            ))
            return 0

        # Run the demo
        console.print(Panel(
            f"[bold cyan]🎬 Launching Milestone {milestone_id} Demo[/bold cyan]\n\n"
            f"[bold]{milestone['emoji']} {milestone['title']}[/bold]\n"
            f"[yellow]Watch your capability in action![/yellow]\n\n"
            f"[cyan]Demonstrating: {milestone['capability']}[/cyan]\n"
            f"[dim]Running: {milestone['demo_file']}[/dim]",
            title="Capability Demo",
            border_style="bright_cyan"
        ))

        try:
            result = subprocess.run(
                [sys.executable, str(demo_path)],
                capture_output=False,
                text=True
            )

            if result.returncode == 0:
                console.print(Panel(
                    f"[bold green]✅ Demo completed successfully![/bold green]\n\n"
                    f"[yellow]You've seen your {milestone['title']} capability in action![/yellow]\n"
                    f"[cyan]Real-world impact: {milestone['real_world_impact']}[/cyan]",
                    title="🎉 Demo Complete",
                    border_style="green"
                ))
            else:
                console.print(f"[yellow]⚠️ Demo completed with status: {result.returncode}[/yellow]")

        except Exception as e:
            console.print(Panel(
                f"[red]❌ Error running demo: {e}[/red]\n\n"
                f"[dim]You can manually run: python capabilities/{milestone['demo_file']}[/dim]",
                title="Demo Error",
                border_style="red"
            ))
            return 1

        return 0

    def _handle_list_command(self, args: Namespace) -> int:
        """Handle milestone list command - show available milestones."""
        console = self.console

        console.print(Panel(
            "[bold cyan]🏆 TinyTorch Milestones[/bold cyan]\n\n"
            "[dim]Recreate ML history from 1957 to 2018[/dim]",
            title="Available Milestones",
            border_style="bright_cyan"
        ))

        # Check module completion status
        # Module workflow saves to progress.json in project root
        progress_file = Path("progress.json")
        completed_modules_raw = []
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    completed_modules_raw = progress_data.get("completed_modules", [])
            except (json.JSONDecodeError, IOError):
                pass

        # Convert completed modules to integers for comparison
        # Handles both "01" and "01_tensor" formats
        completed_module_nums = set()
        for mod in completed_modules_raw:
            try:
                # Extract number from formats like "01" or "01_tensor"
                num_str = mod.split("_")[0] if "_" in mod else mod
                completed_module_nums.add(int(num_str))
            except (ValueError, IndexError):
                pass

        # Check milestone completion
        milestone_progress = self._get_milestone_progress_data()
        completed_milestones = milestone_progress.get("completed_milestones", [])

        for milestone_id in sorted(MILESTONE_SCRIPTS.keys()):
            milestone = MILESTONE_SCRIPTS[milestone_id]

            # Check if prerequisites met (required_modules contains integers)
            prereqs_met = all(mod in completed_module_nums for mod in milestone["required_modules"])
            is_complete = milestone_id in completed_milestones

            # Status indicator
            if is_complete:
                status_icon = "✅"
                status_color = "green"
                status_text = "COMPLETE"
            elif prereqs_met:
                status_icon = "🎯"
                status_color = "yellow"
                status_text = "READY TO RUN"
            else:
                status_icon = "🔒"
                status_color = "dim"
                status_text = "LOCKED"

            # Build display
            if args.simple:
                console.print(f"[{status_color}]{status_icon} {milestone['id']} - {milestone['name']}[/{status_color}]")
            else:
                milestone_display = (
                    f"[{status_color}]{status_icon} {milestone['emoji']} {milestone['name']}[/{status_color}]\n"
                    f"[bold]{milestone['title']}[/bold]\n"
                    f"[dim]{milestone['description']}[/dim]\n"
                    f"[dim]Historical: {milestone['historical_context']}[/dim]\n\n"
                )

                if prereqs_met and not is_complete:
                    milestone_display += f"[bold yellow]▶ Run now:[/bold yellow] [cyan]tito milestone run {milestone_id}[/cyan]\n"
                elif not prereqs_met:
                    missing = [f"{m:02d}" for m in milestone["required_modules"] if m not in completed_module_nums]
                    milestone_display += f"[dim]Required: Complete modules {', '.join(missing)}[/dim]\n"

                console.print(Panel(
                    milestone_display.strip(),
                    title=f"Milestone {milestone['id']} ({milestone['year']})",
                    border_style=status_color
                ))

        return 0

    def _handle_run_command(self, args: Namespace) -> int:
        """Handle milestone run command - run a milestone with checks."""
        console = self.console
        milestone_id = args.milestone_id

        # Resolve name aliases (e.g., "perceptron" -> "01")
        if milestone_id.lower() in MILESTONE_ALIASES:
            milestone_id = MILESTONE_ALIASES[milestone_id.lower()]

        # Validate milestone ID
        if milestone_id not in MILESTONE_SCRIPTS:
            alias_list = ', '.join(sorted(MILESTONE_ALIASES.keys()))
            console.print(Panel(
                f"[red]Invalid milestone: {args.milestone_id}[/red]\n\n"
                f"Valid IDs: {', '.join(sorted(MILESTONE_SCRIPTS.keys()))}\n"
                f"Valid names: {alias_list}",
                title="Invalid Milestone",
                border_style="red"
            ))
            return 1

        milestone = MILESTONE_SCRIPTS[milestone_id]

        # Handle both single script and multiple scripts
        # Also track which script configs we're running (for per-part requirements)
        scripts_to_run = []
        script_configs = []  # Store full config for each script (includes required_modules)

        if "scripts" in milestone:
            all_script_configs = milestone["scripts"]
            all_scripts = [(s["name"], s["script"], s.get("description", "")) for s in all_script_configs]

            # Handle --part flag for multi-part milestones
            if args.part is not None:
                if args.part < 1 or args.part > len(all_scripts):
                    console.print(Panel(
                        f"[red]Invalid part number: {args.part}[/red]\n\n"
                        f"Milestone {milestone_id} has {len(all_scripts)} parts.\n"
                        f"Valid parts: 1-{len(all_scripts)}\n\n"
                        f"[dim]Available parts:[/dim]\n" +
                        "\n".join(f"  Part {i+1}: {s[0]} - {s[2]}" for i, s in enumerate(all_scripts)),
                        title="Invalid Part",
                        border_style="red"
                    ))
                    return 1
                scripts_to_run = [all_scripts[args.part - 1]]
                script_configs = [all_script_configs[args.part - 1]]
                console.print(f"[dim]Running Part {args.part} of {len(all_scripts)}[/dim]\n")
            else:
                scripts_to_run = all_scripts
                script_configs = all_script_configs
        else:
            if args.part is not None:
                console.print(f"[yellow]⚠️ Milestone {milestone_id} has only one part, ignoring --part flag[/yellow]\n")
            scripts_to_run = [("Main", milestone["script"], milestone.get("description", ""))]
            script_configs = [milestone]  # Single script uses milestone-level config

        # Check if all scripts exist
        for script_name, script_file, _ in scripts_to_run:
            script_path = Path(script_file)
            if not script_path.exists():
                console.print(Panel(
                    f"[red]Milestone script not found![/red]\n\n"
                    f"Expected: {script_file}\n"
                    f"[dim]This milestone may not be implemented yet.[/dim]",
                    title="Script Not Found",
                    border_style="red"
                ))
                return 1

        # Check prerequisites and validate exports/tests (unless skipped)
        if not args.skip_checks:
            console.print(f"\n[bold cyan]🔍 Checking prerequisites for Milestone {milestone_id}...[/bold cyan]\n")

            # Check module completion status using module workflow
            from .module.workflow import ModuleWorkflowCommand
            from .src import SrcCommand
            from .test import TestCommand

            module_workflow = ModuleWorkflowCommand(self.config)
            progress_data = module_workflow.get_progress_data()

            source_cmd = SrcCommand(self.config)
            test_cmd = TestCommand(self.config)

            # Determine required modules based on what we're running
            # If running specific part(s), use per-part requirements if available
            # Otherwise use milestone-level requirements
            required_modules = set()
            for config in script_configs:
                part_reqs = config.get('required_modules', milestone.get('required_modules', []))
                required_modules.update(part_reqs)
            required_modules = sorted(required_modules)

            completed_modules = progress_data.get('completed_modules', [])

            # Convert completed to set of integers
            completed_set = {int(m) if isinstance(m, str) else m for m in completed_modules}
            missing_modules = [m for m in required_modules if m not in completed_set]

            if missing_modules:
                part_info = ""
                if args.part is not None and len(script_configs) == 1:
                    part_info = f" (Part {args.part})"
                console.print(Panel(
                    f"[bold yellow]❌ Missing Required Modules[/bold yellow]\n\n"
                    f"[yellow]Milestone {milestone_id}{part_info} requires modules: {', '.join(f'{m:02d}' for m in required_modules)}[/yellow]\n"
                    f"[red]Missing: {', '.join(f'{m:02d}' for m in missing_modules)}[/red]\n\n"
                    f"[cyan]Complete the missing modules first:[/cyan]\n" +
                    "\n".join(f"[dim]  tito module complete {m:02d}[/dim]" for m in missing_modules[:3]),
                    title="Prerequisites Not Met",
                    border_style="yellow"
                ))
                return 1

            console.print(f"[green]✅ All required modules completed![/green]\n")

            # Test imports work
            console.print("[bold cyan]🧪 Testing YOUR implementations...[/bold cyan]\n")

            # Try importing key components (basic check)
            try:
                import sys as _sys
                _sys.path.insert(0, str(Path.cwd()))

                if 1 in milestone["required_modules"]:
                    from tinytorch import Tensor
                    console.print("  [green]✓[/green] Tensor import successful")

                if 2 in milestone["required_modules"]:
                    from tinytorch import ReLU
                    console.print("  [green]✓[/green] Activations import successful")

                if 3 in milestone["required_modules"]:
                    from tinytorch import Linear
                    console.print("  [green]✓[/green] Layers import successful")

                console.print(f"\n[green]✅ YOUR Tiny🔥Torch is ready![/green]\n")

            except ImportError as e:
                console.print(Panel(
                    f"[red]Import Error![/red]\n\n"
                    f"[yellow]Error: {e}[/yellow]\n\n"
                    f"[dim]Your modules may not be exported correctly.[/dim]\n"
                    f"[dim]Try re-exporting: tito module complete XX[/dim]",
                    title="Import Test Failed",
                    border_style="red"
                ))
                return 1

        # Show milestone banner
        scripts_info = ""
        if len(scripts_to_run) > 1:
            scripts_info = "[bold]📂 Parts:[/bold]\n" + "\n".join(
                f"  • {name}: {desc}" for name, _, desc in scripts_to_run
            )
        else:
            scripts_info = f"[bold]📂 Running:[/bold] {scripts_to_run[0][1]}"

        console.print(Panel(
            f"[bold magenta]╔════════════════════════════════════════════════╗[/bold magenta]\n"
            f"[bold magenta]║[/bold magenta]  {milestone['emoji']} Milestone {milestone_id}: {milestone['name']:<30} [bold magenta]║[/bold magenta]\n"
            f"[bold magenta]║[/bold magenta]  {milestone['title']:<44} [bold magenta]║[/bold magenta]\n"
            f"[bold magenta]╚════════════════════════════════════════════════╝[/bold magenta]\n\n"
            f"[bold]📚 Historical Context:[/bold]\n"
            f"{milestone['historical_context']}\n\n"
            f"[bold]🎯 What You'll Do:[/bold]\n"
            f"{milestone['description']}\n\n"
            f"{scripts_info}\n\n"
            f"[dim]All code uses YOUR Tiny🔥Torch implementations![/dim]",
            title=f"🏆 Milestone {milestone_id} ({milestone['year']})",
            border_style="bright_magenta",
            padding=(1, 2)
        ))

        # Only prompt if in interactive terminal
        import sys
        if sys.stdin.isatty() and sys.stdout.isatty():
            try:
                input("\n[yellow]Press Enter to begin...[/yellow] ")
            except EOFError:
                pass

        # Run all milestone scripts
        all_passed = True
        for part_idx, (script_name, script_file, script_desc) in enumerate(scripts_to_run):
            if len(scripts_to_run) > 1:
                console.print(f"\n[bold cyan]━━━ Part {part_idx + 1}/{len(scripts_to_run)}: {script_name} ━━━[/bold cyan]")
                if script_desc:
                    console.print(f"[dim]{script_desc}[/dim]\n")
            else:
                console.print(f"\n[bold green]🚀 Starting Milestone {milestone_id}...[/bold green]\n")

            console.print("━" * 80 + "\n")

            try:
                result = subprocess.run(
                    [sys.executable, script_file],
                    capture_output=False,
                    text=True
                )

                console.print("\n" + "━" * 80)

                if result.returncode != 0:
                    all_passed = False
                    console.print(f"[yellow]⚠️ Part {script_name} completed with errors[/yellow]")
                    if len(scripts_to_run) > 1:
                        # Ask if they want to continue (only in interactive mode)
                        if sys.stdin.isatty() and sys.stdout.isatty():
                            try:
                                cont = input("\n[yellow]Continue to next part? (y/n): [/yellow] ")
                                if cont.lower() != 'y':
                                    return result.returncode
                            except EOFError:
                                return result.returncode
                        else:
                            # Non-interactive: stop on first failure
                            return result.returncode

            except KeyboardInterrupt:
                console.print(f"\n\n[yellow]⚠️ Milestone interrupted by user[/yellow]")
                return 130
            except Exception as e:
                console.print(f"[red]Error running {script_name}: {e}[/red]")
                all_passed = False

        if all_passed:
            # Success! Mark milestone as complete
            self._mark_milestone_complete(milestone_id)

            parts_text = ""
            if len(scripts_to_run) > 1:
                parts_text = f"\n\n[bold]All {len(scripts_to_run)} parts completed:[/bold]\n" + "\n".join(
                    f"  ✅ {name}" for name, _, _ in scripts_to_run
                )

            console.print(Panel(
                f"[bold green]🏆 MILESTONE ACHIEVED![/bold green]\n\n"
                f"[green]You completed Milestone {milestone_id}: {milestone['name']}[/green]\n"
                f"[yellow]{milestone['title']}[/yellow]{parts_text}\n\n"
                f"[bold]What makes this special:[/bold]\n"
                f"• Every line of code: YOUR implementations\n"
                f"• Every tensor operation: YOUR Tensor class\n"
                f"• Every gradient: YOUR autograd\n\n"
                f"[cyan]Achievement saved locally![/cyan]",
                title="✨ Achievement Unlocked ✨",
                border_style="bright_green",
                padding=(1, 2)
            ))

            # Offer to sync progress (uses centralized SubmissionHandler)
            self._offer_progress_sync(milestone_id, milestone['name'])

            # Show next steps
            next_id = str(int(milestone_id) + 1).zfill(2)
            if next_id in MILESTONE_SCRIPTS:
                next_milestone = MILESTONE_SCRIPTS[next_id]
                console.print(f"\n[bold yellow]🎯 What's Next:[/bold yellow]")
                console.print(f"[dim]Milestone {next_id}: {next_milestone['name']} ({next_milestone['year']})[/dim]")

                # Get completed modules for checking next milestone
                progress_file = Path(".tito") / "progress.json"
                completed_modules = []
                if progress_file.exists():
                    try:
                        with open(progress_file, 'r') as f:
                            progress_data = json.load(f)
                            for mod in progress_data.get("completed_modules", []):
                                try:
                                    completed_modules.append(int(mod.split("_")[0]))
                                except (ValueError, IndexError):
                                    pass
                    except (json.JSONDecodeError, IOError):
                        pass

                # Check if unlocked
                missing = [m for m in next_milestone["required_modules"] if m not in completed_modules]
                if missing:
                    console.print(f"[dim]Unlock by completing modules: {', '.join(f'{m:02d}' for m in missing[:3])}[/dim]")
                else:
                    console.print(f"[green]Ready to run: tito milestone run {next_id}[/green]")

            return 0
        else:
            console.print(f"[yellow]⚠️ Milestone completed with errors[/yellow]")
            return 1

    def _handle_info_command(self, args: Namespace) -> int:
        """Handle milestone info command - show detailed information."""
        console = self.console
        milestone_id = args.milestone_id

        # Resolve name aliases (e.g., "perceptron" -> "01")
        if milestone_id.lower() in MILESTONE_ALIASES:
            milestone_id = MILESTONE_ALIASES[milestone_id.lower()]

        if milestone_id not in MILESTONE_SCRIPTS:
            alias_list = ', '.join(sorted(MILESTONE_ALIASES.keys()))
            console.print(Panel(
                f"[red]Invalid milestone: {args.milestone_id}[/red]\n\n"
                f"Valid IDs: {', '.join(sorted(MILESTONE_SCRIPTS.keys()))}\n"
                f"Valid names: {alias_list}",
                title="Invalid Milestone",
                border_style="red"
            ))
            return 1

        milestone = MILESTONE_SCRIPTS[milestone_id]

        # Check status
        progress_file = Path(".tito") / "progress.json"
        completed_modules = []
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    completed_modules = progress_data.get("completed_modules", [])
            except:
                pass

        prereqs_met = all(f"{m:02d}" in completed_modules for m in milestone["required_modules"])

        # Display detailed info
        info_text = (
            f"[bold cyan]{milestone['emoji']} {milestone['name']} ({milestone['year']})[/bold cyan]\n\n"
            f"[bold]{milestone['title']}[/bold]\n\n"
            f"[yellow]📚 Historical Context:[/yellow]\n"
            f"{milestone['historical_context']}\n\n"
            f"[yellow]🎯 Description:[/yellow]\n"
            f"{milestone['description']}\n\n"
            f"[yellow]📋 Required Modules:[/yellow]\n"
        )

        for mod in milestone["required_modules"]:
            mod_str = f"{mod:02d}"
            if mod_str in completed_modules:
                info_text += f"  [green]✓[/green] Module {mod_str}\n"
            else:
                info_text += f"  [red]✗[/red] Module {mod_str}\n"

        # Show scripts
        if "scripts" in milestone:
            info_text += f"\n[yellow]📂 Scripts ({len(milestone['scripts'])} parts):[/yellow]\n"
            for s in milestone["scripts"]:
                info_text += f"  • {s['name']}: {s['script']}\n"
        else:
            info_text += f"\n[yellow]📂 Script:[/yellow] {milestone['script']}\n"

        if prereqs_met:
            info_text += f"\n[bold green]✅ Ready to run![/bold green]\n[cyan]tito milestone run {milestone_id}[/cyan]"
        else:
            missing = [m for m in milestone["required_modules"] if m not in completed_modules]
            info_text += f"\n[bold yellow]🔒 Locked[/bold yellow]\nComplete modules: {', '.join(f'{m:02d}' for m in missing)}"

        console.print(Panel(
            info_text,
            title=f"Milestone {milestone_id} Information",
            border_style="bright_cyan",
            padding=(1, 2)
        ))

        return 0

    def _mark_milestone_complete(self, milestone_id: str) -> None:
        """Mark a milestone as complete in progress tracking."""
        progress = self._get_milestone_progress_data()

        # Add to completed_milestones
        if milestone_id not in progress.get("completed_milestones", []):
            if "completed_milestones" not in progress:
                progress["completed_milestones"] = []
            progress["completed_milestones"].append(milestone_id)
            progress["completion_dates"] = progress.get("completion_dates", {})
            progress["completion_dates"][milestone_id] = datetime.now().isoformat()

        # Also add to unlocked_milestones (for status display)
        if milestone_id not in progress.get("unlocked_milestones", []):
            if "unlocked_milestones" not in progress:
                progress["unlocked_milestones"] = []
            progress["unlocked_milestones"].append(milestone_id)
            progress["unlock_dates"] = progress.get("unlock_dates", {})
            progress["unlock_dates"][milestone_id] = datetime.now().isoformat()
            progress["total_unlocked"] = len(progress["unlocked_milestones"])

        self._save_milestone_progress_data(progress)

    def _get_milestone_progress_data(self) -> dict:
        """Get or create milestone progress data."""
        progress_dir = Path(".tito")
        progress_file = progress_dir / "milestones.json"

        progress_dir.mkdir(exist_ok=True)

        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return {
            "completed_milestones": [],
            "completion_dates": {},
            "unlocked_milestones": [],
            "unlock_dates": {},
            "total_unlocked": 0,
            "achievements": []
        }

    def _save_milestone_progress_data(self, milestone_data: dict) -> None:
        """Save milestone progress data."""
        progress_dir = Path(".tito")
        progress_file = progress_dir / "milestones.json"

        progress_dir.mkdir(exist_ok=True)

        try:
            with open(progress_file, 'w') as f:
                json.dump(milestone_data, f, indent=2)
        except IOError:
            pass

    def _offer_progress_sync(self, milestone_id: str, milestone_name: str) -> None:
        """
        Offer to sync progress after milestone completion.
        Uses the centralized SubmissionHandler for all progress syncing.
        """
        from ..core import auth
        from ..core.submission import SubmissionHandler
        from rich.prompt import Confirm

        console = self.console

        # Check if user is logged in
        if auth.is_logged_in():
            console.print()
            should_sync = Confirm.ask(
                f"[cyan]Would you like to sync this achievement to your profile?[/cyan]",
                default=True
            )

            if should_sync:
                try:
                    # Use the centralized SubmissionHandler
                    handler = SubmissionHandler(self.config, console)

                    # Sync progress (includes modules and milestones)
                    # The handler reads from both progress.json and .tito/milestones.json
                    handler.sync_progress()

                    console.print(f"[green]✅ Milestone {milestone_id} synced to your profile![/green]")
                except Exception as e:
                    console.print(f"[yellow]⚠️ Could not sync: {e}[/yellow]")
                    console.print("[dim]Your progress is saved locally and will sync next time.[/dim]")
        else:
            console.print()
            console.print("[dim]💡 Run 'tito login' to sync your achievements to the leaderboard![/dim]")
