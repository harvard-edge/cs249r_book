import typer
from mlsysim.cli.exceptions import ExitCode, exit_with_code, error_shield
from mlsysim.cli.renderers import render_zoo_table
from mlsysim.hardware.registry import Hardware
from mlsysim.models.registry import Models

def zoo_main(
    ctx: typer.Context,
    category: str = typer.Argument(None, help="Category to explore: 'hardware' or 'models'")
):
    """Explore the built-in registries (The MLSys Zoo)."""
    if not category:
        return
        
    # We retrieve the state from the main app context
    output_format = ctx.obj.get("output_format", "text") if ctx.obj else "text"
    
    if category.lower() == "hardware":
        items = Hardware.list(sort_by="compute.peak_flops", reverse=True)
        render_zoo_table("hardware", items, output_format)
    elif category.lower() == "models":
        items = Models.list(sort_by="parameters", reverse=True)
        render_zoo_table("models", items, output_format)
    else:
        with error_shield(output_format=output_format):
            raise ValueError(f"Unknown zoo category: '{category}'. Valid options are 'hardware' or 'models'.")
    
    exit_with_code(ExitCode.SUCCESS)
