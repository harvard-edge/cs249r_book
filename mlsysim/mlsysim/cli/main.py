import typer

from mlsysim.cli.commands.zoo import zoo_main
from mlsysim.cli.commands.eval import evaluate_main
from mlsysim.cli.commands.serve import serve_main
from mlsysim.cli.commands.schema import schema_main
from mlsysim.cli.commands.optimize import optimize_app
from mlsysim.cli.commands.audit import audit_main

app = typer.Typer(
    name="mlsysim", 
    help="""
    **The ML Systems Infrastructure Modeling Engine.**
    
    A first-principles analytical framework for predicting performance, cost, and carbon footprint of ML workloads.
    """,
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="markdown"
)

# Global state for machine-readable mode
state = {"output_format": "text"}

@app.callback()
def main(ctx: typer.Context, output: str = typer.Option("text", "--output", "-o", help="Output format (text, json, markdown, html)")):
    """Global configuration for the CLI."""
    state["output_format"] = output.lower()
    ctx.obj = {"output_format": state["output_format"]}

# Register modular sub-commands
app.command(name="zoo", help="Explore the built-in registries (The MLSys Zoo).")(zoo_main)
app.command(name="schema", help="Export the JSON Schema for the mlsys.yaml configuration file (for AI agents & IDEs).")(schema_main)
app.command(name="eval", help="[Tier 1] Evaluate the analytical physics of an ML system (via YAML or CLI flags).")(evaluate_main)
app.command(name="serve", help="Evaluate LLM serving performance (prefill + decode).")(serve_main)
app.command(name="audit", help="[Audit] Profile your local hardware against the Iron Law.")(audit_main)
app.add_typer(optimize_app, name="optimize")

if __name__ == "__main__":
    app()
