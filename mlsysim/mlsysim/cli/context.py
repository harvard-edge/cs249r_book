"""Shared CLI context helpers."""

from __future__ import annotations

from typing import Optional

import typer


SUPPORTED_OUTPUT_FORMATS = ("text", "json", "markdown", "html")
OUTPUT_FORMAT_HELP = "Output format (text, json, markdown; html where supported)"


def validate_output_format(value: Optional[str], supported: Optional[set[str]] = None) -> str:
    """Normalize and validate an output format, exiting with a CLI error on mismatch."""
    output_format = (value or "text").lower()
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        typer.echo(
            f"Bad input: unsupported output format '{output_format}'. "
            f"Supported formats: {', '.join(SUPPORTED_OUTPUT_FORMATS)}.",
            err=True,
        )
        raise typer.Exit(1)

    if supported is not None and output_format not in supported:
        typer.echo(
            f"Bad input: output format '{output_format}' is not supported by this command. "
            f"Supported formats: {', '.join(sorted(supported))}.",
            err=True,
        )
        raise typer.Exit(1)

    return output_format


def resolve_output_format(
    ctx: typer.Context,
    output: Optional[str] = None,
    supported: Optional[set[str]] = None,
) -> str:
    """Resolve command-local output override, falling back to global state."""
    if output:
        return validate_output_format(output, supported=supported)
    if ctx.obj:
        return validate_output_format(ctx.obj.get("output_format", "text"), supported=supported)
    return validate_output_format("text", supported=supported)
