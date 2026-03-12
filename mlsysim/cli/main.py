import typer

from mlsysim.cli.commands.zoo import zoo_app
from mlsysim.cli.commands.eval import eval_app
from mlsysim.cli.commands.schema import schema_app
from mlsysim.cli.commands.optimize import optimize_app

app = typer.Typer(
    name="mlsysim", 
    help="The ML Systems Infrastructure Modeling Engine.",
    no_args_is_help=True,
    add_completion=True
)

# Global state for machine-readable mode
state = {"output_format": "text"}

@app.callback()
def main(output: str = typer.Option("text", "--output", "-o", help="Output format (text, json)")):
    """Global configuration for the CLI."""
    state["output_format"] = output.lower()

# Register modular sub-commands
app.add_typer(zoo_app, name="zoo")
app.add_typer(schema_app, name="schema")
app.add_typer(eval_app, name="eval")
app.add_typer(optimize_app, name="optimize")

if __name__ == "__main__":
    app()
