"""
Logo command for TinyTorch CLI: explains the symbolism and meaning behind TinyTorch.
"""

from argparse import ArgumentParser, Namespace
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from pathlib import Path

from .base import BaseCommand

class LogoCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "logo"

    @property
    def description(self) -> str:
        return "Learn about the TinyTorch logo and its meaning"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--image", action="store_true",
                          help="Show path to the actual logo image file")

    def run(self, args: Namespace) -> int:
        console = self.console

        # Display the ASCII logo first
        from ..core.console import print_ascii_logo
        print_ascii_logo()

        # Create the explanation text
        explanation = Text()

        # Title
        explanation.append("\n🔥 The TinyTorch Story\n\n", style="bold yellow")

        # The flame and sparks
        explanation.append("The Flame 🔥\n", style="bold orange1")
        explanation.append(
            "The flame represents the spark of understanding - how learning ML systems "
            "starts with a small flame that can grow into mastery. Just as a torch "
            "lights the way in darkness, TinyTorch illuminates the path to understanding "
            "neural networks from first principles.\n\n",
            style="dim"
        )

        # The sparks
        explanation.append("The Sparks ✨\n", style="bold orange1")
        explanation.append(
            "In our full logo, sparks fly from the flame - representing how knowledge "
            "spreads. Each concept you learn sends off sparks that ignite understanding "
            "in other areas. What starts small can catch fire and grow into something powerful.\n\n",
            style="dim"
        )

        # The "tiny" philosophy
        explanation.append("Why 'Tiny'? \n", style="bold cyan")
        explanation.append(
            "Inspired by TinyML's philosophy: small systems are accessible and powerful. "
            "'Tiny' means we start with the fundamentals, building understanding piece by piece. "
            "By keeping things small and focused, complex concepts become approachable. "
            "Every giant neural network started with tiny building blocks - tensors, gradients, "
            "and simple operations.\n\n",
            style="dim"
        )

        # The "Torch" connection
        explanation.append("Why 'Torch'? \n", style="bold cyan")
        explanation.append(
            "A tribute to PyTorch, the framework that revolutionized deep learning. "
            "While PyTorch is powerful, its complexity can overwhelm beginners. "
            "TinyTorch distills those same concepts into their essence, letting you "
            "build and understand every component. You're not just using a framework - "
            "you're building one.\n\n",
            style="dim"
        )

        # The neural network in the flame
        explanation.append("The Hidden Network 🔥\n", style="bold orange1")
        explanation.append(
            "Look closely at our logo - within the flame, there's a neural network pattern. "
            "This represents the core truth: inside every ML system, no matter how complex, "
            "are simple, connected components working together. The flame contains the network, "
            "just as understanding contains mastery.\n\n",
            style="dim"
        )

        # The philosophy
        explanation.append("The Philosophy 💡\n", style="bold yellow")
        explanation.append(
            "TinyTorch embodies the belief that anyone can understand ML systems by building them. "
            "Start small, understand deeply, build everything. What begins as a tiny flame of "
            "curiosity becomes the torch that lights your path to ML engineering mastery.\n\n",
            style="dim"
        )

        # Personal message from the creator
        explanation.append("\n— ", style="dim")
        explanation.append("Professor Vijay Janapa Reddi\n", style="bold cyan")
        explanation.append("   Harvard University • CS 249R: Tiny Machine Learning\n\n", style="dim")

        # Personal message - flows naturally, only the tagline pops
        explanation.append(
            "TinyTorch grew from the TinyML movement that started at Harvard in CS 249R. As I taught "
            "thousands of students, both online and in person, I realized something was missing: there "
            "was no way to truly engineer ML systems from the ground up. So that's why I built "
            "TinyTorch—a hands-on companion to ",
            style="italic dim"
        )
        explanation.append("www.mlsysbook.ai", style="italic magenta")
        explanation.append(
            ", where you don't just import libraries, you engineer every component yourself. "
            "Think of it as building the Lego blocks for your favorite Star Wars set.\n\n",
            style="italic dim"
        )
        explanation.append("So don't just ", style="bold cyan")
        explanation.append("\"import torch\"", style="white")
        explanation.append(", build it.", style="bold cyan")
        explanation.append("\n\n", style="dim")

        # Display in a nice panel
        console.print(Panel(
            explanation,
            title="[bold]About the TinyTorch Logo[/bold]",
            border_style="orange1",
            padding=(1, 2)
        ))

        # Show logo file path if requested
        if args.image:
            logo_path = Path(__file__).parent.parent.parent / "logo" / "logo.png"
            if logo_path.exists():
                console.print(f"\n[cyan]Logo image location:[/cyan] {logo_path}")
                console.print("[dim]Open this file to see the full logo with sparks[/dim]")
            else:
                console.print(f"\n[yellow]Logo image not found at expected location[/yellow]")

        # Final inspiring message
        console.print("\n[yellow]🔥 [/yellow][italic]Start tiny. Go deep. Build big.[/italic]\n")

        return 0
