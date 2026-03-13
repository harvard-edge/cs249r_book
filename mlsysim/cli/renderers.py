import json
from typing import Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Strict I/O Purity (Guideline 1)
console_err = Console(stderr=True) # All diagnostics, warnings, UI go to stderr
console_out = Console()            # stdout is reserved for the final payload

def print_error(title: str, message: str, output_format: str = "text"):
    """Renders a semantic error box, or a JSON error."""
    if output_format == "json":
        # If in JSON mode, we still print the error to stdout so the script can parse the failure reason
        print(json.dumps({"status": "error", "title": title, "reason": message}))
    elif output_format == "markdown":
        console_err.print(f"**🚨 {title}**\n\n{message}")
    else:
        console_err.print(Panel(message, title=f"[bold red]🚨 {title}[/bold red]", border_style="red"))

def print_warning(message: str):
    """Warnings ALWAYS go to stderr so they don't break JSON pipes."""
    console_err.print(f"[bold yellow]⚠️ WARNING:[/bold yellow] {message}")

def render_zoo_table(registry_name: str, items: list, output_format: str):
    """Renders a registry listing (Zoo) as a table or JSON."""
    if output_format == "json":
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
    elif output_format == "markdown":
        print(f"## The MLSys {registry_name.title()} Zoo\n")
        if items and hasattr(items[0], "compute"):
            print("| Hardware Name | Peak FLOP/s | HBM Bandwidth | TDP |")
            print("|---|---|---|---|")
            for item in items:
                tdp_str = str(item.tdp) if item.tdp else "N/A"
                print(f"| {item.name} | {item.compute.peak_flops:~P} | {item.memory.bandwidth:~P} | {tdp_str} |")
        elif items and hasattr(items[0], "parameters"):
            print("| Model Name | Architecture | Parameters | Layers |")
            print("|---|---|---|---|")
            for item in items:
                print(f"| {item.name} | {item.architecture} | {item.parameters:~P} | {item.layers} |")
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

def render_scorecard(eval_obj, output_format: str):
    """Renders the unified 3-lens SystemEvaluation scorecard."""
    
    def _format_metric(k: str, v: Any) -> str:
        if isinstance(v, float):
            val = f"{v:,.2f}"
            if "latency" in k.lower(): val += " ms"
            elif "throughput" in k.lower(): val += " / s"
            elif "memory" in k.lower() and "gb" in k.lower(): val += " GB"
            elif "mfu" in k.lower(): val = f"{v:.1%}"
            return val
        return f"{v}"

    if output_format == "json":
        # Machine Mode: stdout gets strict JSON
        print(json.dumps(eval_obj.to_dict(), indent=2))
        return
    elif output_format == "markdown":
        print(f"## MLSys·im Plan: {eval_obj.scenario_name}")
        
        f_icon = "🟢" if eval_obj.feasibility.status == "PASS" else "🔴"
        print(f"\n### {f_icon} Feasibility: {eval_obj.feasibility.status}")
        print(f"{eval_obj.feasibility.summary}")
        
        print(f"\n### 🚀 Performance: {eval_obj.performance.status}")
        print(f"{eval_obj.performance.summary}")
        for k, v in eval_obj.performance.metrics.items():
            print(f"- **{k.replace('_', ' ').title()}**: {_format_metric(k, v)}")
            
        if eval_obj.macro.status != "SKIPPED":
            print(f"\n### 🌍 Ops & Macro: {eval_obj.macro.status}")
            print(f"{eval_obj.macro.summary}")
            for k, v in eval_obj.macro.metrics.items():
                print(f"- **{k.replace('_', ' ').title()}**: {_format_metric(k, v)}")
        return
    elif output_format == "html":
        # Generate a beautiful standalone HTML dashboard
        f_icon = "🟢" if eval_obj.feasibility.status == "PASS" else "🔴"
        f_color = "#10b981" if eval_obj.feasibility.status == "PASS" else "#ef4444"
        
        perf_html = "".join([f"<tr><td style='padding:8px; border-bottom:1px solid #eee;'><b>{k.replace('_', ' ').title()}</b></td><td style='padding:8px; border-bottom:1px solid #eee; text-align:right;'>{_format_metric(k, v)}</td></tr>" for k, v in eval_obj.performance.metrics.items()])
        
        macro_html = ""
        if eval_obj.macro.status != "SKIPPED":
            macro_rows = "".join([f"<tr><td style='padding:8px; border-bottom:1px solid #eee;'><b>{k.replace('_', ' ').title()}</b></td><td style='padding:8px; border-bottom:1px solid #eee; text-align:right;'>{_format_metric(k, v)}</td></tr>" for k, v in eval_obj.macro.metrics.items()])
            macro_html = f"""
            <div class="card" style="border-left: 4px solid #8b5cf6;">
                <h3 style="color:#8b5cf6;">🌍 Ops & Macro: {eval_obj.macro.status}</h3>
                <p>{eval_obj.macro.summary}</p>
                <table style="width:100%; border-collapse:collapse; font-size:14px; margin-top:15px;">{macro_rows}</table>
            </div>
            """
            
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>MLSys·im Plan: {eval_obj.scenario_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f9fafb; color: #111827; padding: 40px; line-height: 1.5; }}
        .container {{ max-width: 600px; margin: 0 auto; }}
        .card {{ background: white; padding: 24px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        h2 {{ color: #1f2937; text-align: center; margin-bottom: 30px; }}
        h3 {{ margin-top: 0; margin-bottom: 8px; font-size: 18px; }}
        p {{ margin-top: 0; color: #4b5563; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>🚀 MLSys·im Plan<br><span style="color:#6b7280; font-size:18px; font-weight:normal;">{eval_obj.scenario_name}</span></h2>
        
        <div class="card" style="border-left: 4px solid {f_color};">
            <h3 style="color:{f_color};">{f_icon} Feasibility: {eval_obj.feasibility.status}</h3>
            <p>{eval_obj.feasibility.summary}</p>
        </div>
        
        <div class="card" style="border-left: 4px solid #3b82f6;">
            <h3 style="color:#3b82f6;">⚡ Performance: {eval_obj.performance.status}</h3>
            <p>{eval_obj.performance.summary}</p>
            <table style="width:100%; border-collapse:collapse; font-size:14px; margin-top:15px;">
                {perf_html}
            </table>
        </div>
        
        {macro_html}
    </div>
</body>
</html>"""
        print(html_content)
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
        table.add_row(f"  • {k.replace('_', ' ').title()}", _format_metric(k, v))
            
    # Tier 3: Macro (Optional)
    if eval_obj.macro.status != "SKIPPED":
        m_color = "magenta"
        table.add_row("", "")
        table.add_row(f"[bold {m_color}]🌍 Ops & Macro: {eval_obj.macro.status}[/bold {m_color}]", f"({eval_obj.macro.summary})")
        for k, v in eval_obj.macro.metrics.items():
            table.add_row(f"  • {k.replace('_', ' ').title()}", _format_metric(k, v))
    
    panel = Panel(
        table,
        title=f"[bold]MLSys·im Plan: {eval_obj.scenario_name}[/bold]",
        border_style="green" if eval_obj.passed_all else "red",
        expand=False
    )
    
    console_out.print(panel)


def render_optimization(opt_name: str, opt_result, output_format: str):
    """Renders the output of any OptimizerResult."""
    if output_format == "json":
        print(json.dumps(opt_result.model_dump(mode="json"), indent=2))
        return
    elif output_format == "markdown":
        print(f"## MLSys·im Optimize: {opt_name}")
        print(f"\n### 🎯 Objective Value: {opt_result.objective_value:,.2f}")
        print("\n### 🏆 Best Configuration")
        for k, v in opt_result.best_config.items():
            print(f"- **{k.replace('_', ' ').title()}**: {v}")
        print(f"\n*Searched {opt_result.total_searched} configurations.*")
        return
    elif output_format == "html":
        config_html = "".join([f"<tr><td style='padding:8px; border-bottom:1px solid #eee;'><b>{k.replace('_', ' ').title()}</b></td><td style='padding:8px; border-bottom:1px solid #eee; text-align:right;'>{v}</td></tr>" for k, v in opt_result.best_config.items()])
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>MLSys·im Optimize: {opt_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f9fafb; color: #111827; padding: 40px; line-height: 1.5; }}
        .container {{ max-width: 600px; margin: 0 auto; }}
        .card {{ background: white; padding: 24px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        h2 {{ color: #1f2937; text-align: center; margin-bottom: 30px; }}
        h3 {{ margin-top: 0; margin-bottom: 8px; font-size: 18px; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>🔍 MLSys·im Optimize<br><span style="color:#6b7280; font-size:18px; font-weight:normal;">{opt_name}</span></h2>
        
        <div class="card" style="border-left: 4px solid #06b6d4; text-align: center;">
            <h3 style="color:#06b6d4; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">🎯 Objective Value</h3>
            <div style="font-size: 32px; font-weight: bold; color: #111827;">{opt_result.objective_value:,.2f}</div>
        </div>
        
        <div class="card" style="border-left: 4px solid #10b981;">
            <h3 style="color:#10b981;">🏆 Best Configuration</h3>
            <table style="width:100%; border-collapse:collapse; font-size:14px; margin-top:15px; margin-bottom: 15px;">
                {config_html}
            </table>
            <p style="color: #6b7280; font-size: 12px; margin-bottom: 0; text-align: center;">
                <i>Searched {opt_result.total_searched} configurations.</i>
            </p>
        </div>
    </div>
</body>
</html>"""
        print(html_content)
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
