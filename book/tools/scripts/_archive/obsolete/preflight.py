import subprocess
import sys

from rich.console import Console


def run_pre_commit():
    """
    Run all pre-commit hooks and report the status.
    """
    console = Console()
    console.print("[bold blue]🚀 Running pre-commit checks...[/bold blue]")

    try:
        process = subprocess.run(
            ["pre-commit", "run", "--all-files"],
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit
        )

        if process.returncode == 0:
            console.print("[bold green]✅ Pre-commit checks passed successfully![/bold green]")
            return True
        else:
            console.print("[bold red]❌ Pre-commit checks failed.[/bold red]")
            console.print("\n[yellow]Output:[/yellow]")
            console.print(process.stdout)
            console.print(process.stderr)
            return False
    except FileNotFoundError:
        console.print("[bold red]Error: 'pre-commit' command not found.[/bold red]")
        console.print("Please ensure pre-commit is installed and in your PATH.")
        return False
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        return False


def main():
    """
    Main function to run all pre-flight checks.
    """
    console = Console()
    console.print("[bold magenta]✈️  Starting pre-flight checks...[/bold magenta]\n")

    pre_commit_ok = run_pre_commit()

    if not pre_commit_ok:
        sys.exit(1)

    # Future checks can be added here
    # For example:
    # slow_tests_ok = run_slow_tests()
    # if not slow_tests_ok:
    #     sys.exit(1)

    console.print("\n[bold green]🎉 All pre-flight checks passed![/bold green]")
    sys.exit(0)


if __name__ == "__main__":
    main()
