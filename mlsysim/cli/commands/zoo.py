import typer
from mlsysim.cli.exceptions import ExitCode, exit_with_code, error_shield
from mlsysim.cli.renderers import render_zoo_table
from mlsysim.hardware.registry import Hardware
from mlsysim.models.registry import Models

zoo_app = typer.Typer(help="Explore the built-in registries (The MLSys Zoo).", no_args_is_help=True)

@zoo_app.callback(invoke_without_command=True)
def zoo_main(
    ctx: typer.Context,
    category: str = typer.Argument(None, help="Category to explore: 'hardware' or 'models'")
):
    """Explore the built-in registries (The MLSys Zoo)."""
    if not category:
        return
        
    # We retrieve the state from the main app context
    output_format = ctx.obj.get("output_format", "text") if ctx.obj else "text"
    is_json = output_format == "json"
    
    if category.lower() == "hardware":
        items = Hardware.list(sort_by="compute.peak_flops", reverse=True)
        render_zoo_table("hardware", items, is_json)
    elif category.lower() == "models":
        items = Models.list(sort_by="parameters", reverse=True)
        render_zoo_table("models", items, is_json)
    else:
        with error_shield(output_format=output_format):
            raise ValueError(f"Unknown zoo category: '{category}'. Valid options are 'hardware' or 'models'.")
    
    exit_with_code(ExitCode.SUCCESS)
