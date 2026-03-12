import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Strict I/O Purity (Guideline 1)
console_err = Console(stderr=True) # All diagnostics, warnings, UI go to stderr
console_out = Console()            # stdout is reserved for the final payload

def print_error(title: str, message: str, is_json: bool = False):
    """Renders a semantic error box, or a JSON error."""
    if is_json:
        # If in JSON mode, we still print the error to stdout so the script can parse the failure reason
        print(json.dumps({"status": "error", "title": title, "reason": message}))
    else:
        console_err.print(Panel(message, title=f"[bold red]🚨 {title}[/bold red]", border_style="red"))

def print_warning(message: str):
    """Warnings ALWAYS go to stderr so they don't break JSON pipes."""
    console_err.print(f"[bold yellow]⚠️ WARNING:[/bold yellow] {message}")

def render_zoo_table(registry_name: str, items: list, is_json: bool):
    """Renders a registry listing (Zoo) as a table or JSON."""
    if is_json:
        # Extract primitive data for JSON dump
        data = []
        for item in items:
            row = {"name": item.name}
            if hasattr(item, "compute"):
                row["flops"] = str(item.compute.peak_flops)
                row["bandwidth"] = str(item.memory.bandwidth)
            elif hasattr(item, "parameters"):
                row["parameters"] = str(item.parameters)
                row["layers"] = item.layers
            data.append(row)
        print(json.dumps({registry_name: data}, indent=2))
        return

    table = Table(title=f"The MLSys {registry_name.title()} Zoo", box=None, padding=(0, 2))
    
    if items and hasattr(items[0], "compute"):
        table.add_column("Hardware Name", style="cyan", no_wrap=True)
        table.add_column("Peak FLOP/s", justify="right", style="green")
        table.add_column("HBM Bandwidth", justify="right", style="magenta")
        table.add_column("TDP", justify="right")
        for item in items:
            tdp_str = str(item.tdp) if item.tdp else "N/A"
            table.add_row(item.name, f"{item.compute.peak_flops:~P}", f"{item.memory.bandwidth:~P}", tdp_str)
            
    elif items and hasattr(items[0], "parameters"):
        table.add_column("Model Name", style="cyan", no_wrap=True)
        table.add_column("Architecture", style="white")
        table.add_column("Parameters", justify="right", style="green")
        table.add_column("Layers", justify="right")
        for item in items:
            table.add_row(item.name, item.architecture, f"{item.parameters:~P}", str(item.layers))

    console_out.print(table)

def render_scorecard(eval_obj, is_json: bool):
    """Renders the unified 3-lens SystemEvaluation scorecard."""
    if is_json:
        # Machine Mode: stdout gets strict JSON
        print(json.dumps(eval_obj.to_dict(), indent=2))
        return

    # Human Mode: Render the UI Scorecard
    table = Table(show_header=False, box=None, padding=(0, 2))
    
    # Tier 1: Feasibility
    f_color = "green" if eval_obj.feasibility.status == "PASS" else "red"
    f_icon = "🟢" if eval_obj.feasibility.status == "PASS" else "🔴"
    table.add_row(f"[bold {f_color}]{f_icon} Feasibility: {eval_obj.feasibility.status}[/bold {f_color}]", f"({eval_obj.feasibility.summary})")
    
    # Tier 2: Performance
    p_color = "blue"
    table.add_row(f"[bold {p_color}]🚀 Performance: {eval_obj.performance.status}[/bold {p_color}]", f"({eval_obj.performance.summary})")
    
    for k, v in eval_obj.performance.metrics.items():
        if isinstance(v, float):
            table.add_row(f"  • {k.replace('_', ' ').title()}", f"{v:.2f}")
        else:
            table.add_row(f"  • {k.replace('_', ' ').title()}", f"{v}")
            
    # Tier 3: Macro (Optional)
    if eval_obj.macro.status != "SKIPPED":
        m_color = "magenta"
        table.add_row("", "")
        table.add_row(f"[bold {m_color}]🌍 Ops & Macro: {eval_obj.macro.status}[/bold {m_color}]", f"({eval_obj.macro.summary})")
        for k, v in eval_obj.macro.metrics.items():
            if isinstance(v, float):
                table.add_row(f"  • {k.replace('_', ' ').title()}", f"{v:,.2f}")
            else:
                table.add_row(f"  • {k.replace('_', ' ').title()}", f"{v}")
    
    panel = Panel(
        table,
        title=f"[bold]MLSys·im Plan: {eval_obj.scenario_name}[/bold]",
        border_style="green" if eval_obj.passed_all else "red",
        expand=False
    )
    
    console_out.print(panel)


def render_optimization(opt_name: str, opt_result, is_json: bool):
    """Renders the output of any OptimizerResult."""
    if is_json:
        print(json.dumps(opt_result.model_dump(mode="json"), indent=2))
        return

    table = Table(show_header=False, box=None, padding=(0, 2))
    
    table.add_row("[bold cyan]🎯 Objective Value[/bold cyan]", f"{opt_result.objective_value:,.2f}")
    table.add_row("", "")
    
    table.add_row("[bold green]🏆 Best Configuration[/bold green]", "")
    for k, v in opt_result.best_config.items():
        table.add_row(f"  • {k.replace('_', ' ').title()}", f"{v}")
        
    table.add_row("", "")
    table.add_row(f"[italic gray]Searched {opt_result.total_searched} configurations.[/italic gray]", "")

    panel = Panel(
        table,
        title=f"[bold]MLSys·im Optimize: {opt_name}[/bold]",
        border_style="cyan",
        expand=False
    )
    
    console_out.print(panel)
