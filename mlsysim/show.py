"""
show.py — Tutorial display helpers for MLSys·im.

Replaces verbose print(f"...") patterns with clean, aligned output.
Two primitives: info() for key-value blocks, table() for tabular data.

Usage in tutorials:
    from mlsysim.show import info, table, banner

    info("Phase Analysis",
         TTFT=result.ttft.to('ms'),
         ITL=result.itl.to('ms'),
         Memory=f"{result.memory_utilization:.1%}")

    table(["GPU", "TTFT (ms)", "ITL (ms)"],
          [["A100", 12.3, 0.84],
           ["H100",  6.1, 0.42]])
"""

from .core.constants import ureg


def _format_value(v):
    """Auto-format a value for display."""
    if isinstance(v, ureg.Quantity):
        mag = v.magnitude
        unit_str = f"{v.units:~P}"
        # Choose precision based on magnitude
        if mag == int(mag) and abs(mag) >= 1:
            num = f"{int(mag):,}"
        elif abs(mag) >= 100:
            num = f"{mag:,.1f}"
        elif abs(mag) >= 1:
            num = f"{mag:.2f}"
        elif abs(mag) >= 0.01:
            num = f"{mag:.3f}"
        else:
            num = f"{mag:.2e}"
        return f"{num} {unit_str}"
    if isinstance(v, float):
        if abs(v) >= 1e6:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:,.1f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        return f"{v:.4f}"
    return str(v)


def info(title=None, **fields):
    """Print an aligned key-value block with optional section title.

    Args:
        title: Optional section header (rendered as ── Title ──)
        **fields: Key-value pairs to display. Keys become labels,
                  values are auto-formatted (pint Quantities, floats, strings).

    Example:
        info("Memory Budget",
             Weights=result.model_weights_size,
             KV_cache=result.kv_cache_size,
             Utilization=f"{result.memory_utilization:.1%}")

        Output:
        ── Memory Budget ──────────────────────────
        Weights:       80.00 GB
        KV cache:      2.34 GB
        Utilization:   29.3%
    """
    if title:
        header = f"── {title} "
        print(header + "─" * max(0, 44 - len(header)))

    if not fields:
        return

    # Convert underscores to spaces in keys for display
    display_fields = {k.replace("_", " "): v for k, v in fields.items()}
    width = max(len(k) for k in display_fields) + 1  # +1 for colon
    for key, val in display_fields.items():
        print(f"{key + ':':<{width}}  {_format_value(val)}")


def table(headers, rows, alignments=None):
    """Print a clean aligned table.

    Args:
        headers: List of column header strings.
        rows: List of lists (one per row). Values are auto-formatted.
        alignments: Optional string of '<', '>', '^' per column.
                    Defaults to: first column left, rest right.

    Example:
        table(["GPU", "TTFT (ms)", "ITL (ms)", "Verdict"],
              [["A100", 12.3, 0.84, "✓"],
               ["H100",  6.1, 0.42, "✓"]])
    """
    if not rows:
        return

    n_cols = len(headers)

    if alignments is None:
        alignments = "<" + ">" * (n_cols - 1)
    alignments = alignments.ljust(n_cols, ">")

    # Format all cells
    formatted = []
    for row in rows:
        formatted.append([_format_value(v) for v in row])

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in formatted:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    # Build format strings
    def _align(text, width, align_char):
        if align_char == "<":
            return text.ljust(width)
        elif align_char == "^":
            return text.center(width)
        return text.rjust(width)

    # Print header
    header_line = "  ".join(
        _align(h, widths[i], alignments[i]) for i, h in enumerate(headers)
    )
    print(header_line)
    print("─" * len(header_line))

    # Print rows
    for row in formatted:
        print("  ".join(
            _align(row[i] if i < len(row) else "", widths[i], alignments[i])
            for i in range(n_cols)
        ))


def banner(text):
    """Print a section banner for tutorial output.

    Example:
        banner("Domain: Node (Walls 1-4)")

        Output:
        === Domain: Node (Walls 1-4) ===
    """
    print(f"\n=== {text} ===")
