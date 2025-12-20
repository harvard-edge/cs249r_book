#!/usr/bin/env python3
"""
Module Completion Orchestrator
Runs export, integration tests, and capability demonstrations
"""

import sys
from pathlib import Path
import subprocess
import time
from typing import Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.integration.run_module_tests import ModuleIntegrationTester

# Map modules to their capability demonstrations
CAPABILITY_DEMOS = {
    "03_layers": "capabilities/03_neural_networks/demonstrate.py",
    "09_convolutions": "capabilities/09_computer_vision/demonstrate.py",
    "12_attention": "capabilities/12_attention_mechanism/demonstrate.py",
    "08_training": "capabilities/08_complete_training/demonstrate.py",
    "13_transformers": "capabilities/13_language_model/demonstrate.py",
}

class ModuleCompletionOrchestrator:
    """Orchestrate the complete module completion workflow."""

    def __init__(self):
        self.results = {
            'export': None,
            'integration': None,
            'capability': None
        }

    def complete_module(self, module_name: str, skip_test: bool = False) -> bool:
        """
        Complete workflow for module completion:
        1. Export module to package
        2. Run integration tests
        3. Run capability demonstration (if tests pass)
        """
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

        console = Console()

        # Welcome banner
        console.print("\n" + "="*70)
        console.print(Panel(f"[bold cyan]📦 Module Completion: {module_name}[/bold cyan]",
                          style="bold blue", expand=False))
        console.print("="*70 + "\n")

        # Step 1: Export module
        console.print("[bold blue]Step 1: Exporting Module to Package[/bold blue]")
        export_success = self._export_module(module_name)

        if not export_success:
            console.print("[red]❌ Export failed! Please fix errors and try again.[/red]")
            return False

        console.print("[green]✅ Module exported successfully![/green]\n")

        if skip_test:
            console.print("[yellow]⏭️  Skipping tests (--skip-test flag)[/yellow]")
            return True

        # Step 2: Run integration tests
        console.print("[bold yellow]Step 2: Running Integration Tests[/bold yellow]")
        console.print("[dim]Testing how this module interfaces with its dependencies...[/dim]\n")

        test_success = self._run_integration_tests(module_name)

        if not test_success:
            console.print("\n[red]❌ Integration tests failed![/red]")
            console.print("[dim]Fix the issues above and run again.[/dim]")
            return False

        # Step 3: Run capability demonstration (if available)
        if module_name in CAPABILITY_DEMOS:
            console.print("\n[bold magenta]Step 3: Capability Demonstration[/bold magenta]")
            console.print("[dim]Showing what you can now do with TinyTorch...[/dim]\n")

            demo_success = self._run_capability_demo(module_name)

            if demo_success:
                console.print("\n[green]✨ Capability demonstrated successfully![/green]")
            else:
                console.print("\n[yellow]⚠️ Demo had issues, but module is complete.[/yellow]")
        else:
            console.print("\n[dim]No capability demo for this module.[/dim]")

        # Final success message
        console.print("\n" + "🌟"*35)
        console.print(Panel("[bold green]✅ MODULE COMPLETE![/bold green]\n\n"
                          f"Module {module_name} has been successfully:\n"
                          "• Exported to the TinyTorch package\n"
                          "• Tested for integration with dependencies\n"
                          "• Demonstrated (if applicable)\n\n"
                          "You can now use this module's functionality!",
                          style="green", expand=False))
        console.print("🌟"*35 + "\n")

        # Suggest next module
        self._suggest_next_module(module_name, console)

        return True

    def _export_module(self, module_name: str) -> bool:
        """Export module using nbdev."""
        try:
            # Run nbdev_export for the specific module
            cmd = ["nbdev_export", "--path", f"modules/{module_name}/{module_name}.py"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return True
            else:
                print(f"Export error: {result.stderr}")
                return False
        except Exception as e:
            print(f"Export failed: {e}")
            return False

    def _run_integration_tests(self, module_name: str) -> bool:
        """Run integration tests for the module."""
        tester = ModuleIntegrationTester()
        results = tester.run_module_tests(module_name)
        success = tester.print_report(results)
        self.results['integration'] = results
        return success

    def _run_capability_demo(self, module_name: str) -> bool:
        """Run capability demonstration if available."""
        demo_path = CAPABILITY_DEMOS.get(module_name)
        if not demo_path:
            return True

        demo_file = Path(demo_path)
        if not demo_file.exists():
            print(f"Demo file not found: {demo_file}")
            return False

        try:
            result = subprocess.run(
                [sys.executable, str(demo_file)],
                capture_output=False,  # Let demo print directly
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Demo failed: {e}")
            return False

    def _suggest_next_module(self, current_module: str, console):
        """Suggest the next module to complete."""
        module_order = [
            "01_tensor", "02_activations", "03_layers", "04_losses", "05_dataloader",
            "06_autograd", "07_optimizers", "08_training", "09_convolutions",
            "10_tokenization", "11_embeddings", "12_attention", "13_transformers",
            "14_profiling", "15_quantization", "16_compression", "17_acceleration",
            "18_memoization", "19_benchmarking", "20_capstone"
        ]

        try:
            current_idx = module_order.index(current_module)
            if current_idx < len(module_order) - 1:
                next_module = module_order[current_idx + 1]
                console.print(f"\n[cyan]🚀 Next: tito module complete {next_module}[/cyan]")
        except ValueError:
            pass


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Complete module workflow")
    parser.add_argument("module", help="Module name (e.g., 05_dataloader)")
    parser.add_argument("--skip-test", action="store_true",
                      help="Skip integration tests")

    args = parser.parse_args()

    orchestrator = ModuleCompletionOrchestrator()
    success = orchestrator.complete_module(args.module, args.skip_test)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
