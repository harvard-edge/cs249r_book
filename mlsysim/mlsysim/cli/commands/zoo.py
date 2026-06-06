import typer
from typing import Optional
from mlsysim.cli.context import OUTPUT_FORMAT_HELP, resolve_output_format
from mlsysim.cli.exceptions import ExitCode, exit_with_code, error_shield
from mlsysim.cli.renderers import render_zoo_table
from mlsysim.hardware.registry import Hardware
from mlsysim.models.registry import Models

def zoo_main(
    ctx: typer.Context,
    category: str = typer.Argument(
        None, help="Category to explore: 'hardware' or 'models' (omit to list both)"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help=OUTPUT_FORMAT_HELP),
):
    """Explore the built-in registries (The MLSys Zoo)."""
    output_format = resolve_output_format(ctx, output, supported={"text", "json", "markdown"})

    if not category:
        # The argument is genuinely optional (2026-06-06 UX fix: it used to be
        # marked optional in --help yet error out): no category renders both.
        render_zoo_table("hardware", Hardware.list(sort_by="compute.peak_flops", reverse=True), output_format)
        render_zoo_table("models", Models.list(sort_by="parameters", reverse=True), output_format)
        exit_with_code(ExitCode.SUCCESS)

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
