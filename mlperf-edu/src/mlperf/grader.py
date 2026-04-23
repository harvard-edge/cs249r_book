import os
import json
import glob
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

class GradingEngine:
    """
    Ingests all local `submissions/*.json` payloads dynamically, isolates them by
    Division and Machine Architecture, and constructs the Global Pedagogy Leaderboard.
    """
    def __init__(self, submissions_dir="submissions"):
        self.submissions_dir = submissions_dir
        self.leaderboards = defaultdict(lambda: defaultdict(list))
        # Structure: leaderboards[task][hardware_tier] = [submission1, submission2, ...]

    def parse_submissions(self):
        if not os.path.exists(self.submissions_dir):
            console.print(f"[bold red]❌ No '{self.submissions_dir}' directory found. Have students run their benchmarks first.[/bold red]")
            return False

        files = glob.glob(os.path.join(self.submissions_dir, "*.json"))
        if not files:
            console.print("[yellow]⚠️ Submissions folder exists but no JSON payloads were found.[/yellow]")
            return False

        for f in files:
            try:
                with open(f, "r") as file:
                    data = json.load(file)
                    telemetry = data.get("telemetry", {})
                    sut = data.get("system_under_test", {})
                    
                    # Core mapping
                    task_name = telemetry.get('scenario', 'Unknown_Task')
                    
                    # We normalize hardware generically to compare students on fair ground
                    hw_tier = sut.get('device', 'Unknown_Hardware')
                    
                    # Calculate sorting metric (QPS is higher-is-better, Latency is lower-is-better)
                    metric_score = telemetry.get('queries_per_second', 0.0)
                    
                    self.leaderboards[task_name][hw_tier].append({
                        "file": os.path.basename(f),
                        "timestamp": data.get("metadata", {}).get("timestamp", "N/A"),
                        "division": telemetry.get("division_passed", "FAIL"),
                        "qps": metric_score,
                        "latency_p90": telemetry.get("latency_p90", 0.0),
                        "joules": telemetry.get("estimated_joules", 0.0)
                    })
            except Exception as e:
                console.print(f"[bold red]❌ Corrupted payload detected: {f} ({e})[/bold red]")
        
        return True

    def generate_report(self):
        """Builds terminal-rendering and markdown tables for the professors."""
        if not self.parse_submissions():
            return
            
        console.print(Panel("[bold cyan]🏆 MLPerf EDU Global Leaderboard Engine[/bold cyan]", expand=False))
        
        for task, hardware_tiers in self.leaderboards.items():
            console.print(f"\n[bold green]🏁 Scenario: {task}[/bold green]")
            
            for hw_tier, submissions in hardware_tiers.items():
                console.print(f"  [bold blue]💻 Hardware Sub-Tier: {hw_tier}[/bold blue]")
                
                # Sort descending by QPS
                sorted_subs = sorted(submissions, key=lambda x: x['qps'], reverse=True)
                
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Rank", justify="center")
                table.add_column("SUT Payload Timestamp")
                table.add_column("Division")
                table.add_column("Throughput (QPS)", justify="right")
                table.add_column("p90 Latency (s)", justify="right")
                table.add_column("Energy Burn (Joules)", justify="right")
                
                for idx, sub in enumerate(sorted_subs):
                    rank = f"#{idx+1}"
                    # Highlight disqualified payloads
                    div_style = "green" if sub["division"] != "FAIL" else "red"
                    
                    table.add_row(
                        rank,
                        sub['timestamp'],
                        f"[{div_style}]{sub['division'].upper()}[/{div_style}]",
                        f"{sub['qps']:.2f}",
                        f"{sub['latency_p90']:.4f}",
                        f"{sub['joules']:.1f}"
                    )
                
                console.print(table)
                console.print("")
                
        # Future-proofing: Here the TA can dump this to HTML/Markdown
        console.print("[dim]Grading sweep completely executed natively.[/dim]")

def execute_grading(args):
    grader = GradingEngine()
    grader.generate_report()
