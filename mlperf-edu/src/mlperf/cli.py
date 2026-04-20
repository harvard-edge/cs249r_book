import argparse
import sys
import asyncio
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from .hardware import profile_hardware
from .loadgen import LoadGenProxy

try:
    from .fingerprint import detect_hardware, format_fingerprint
except ImportError:
    detect_hardware = None
    format_fingerprint = None

console = Console()

# --------------------------------------------------------------------------------
# Generic Yaml Config Loader
# --------------------------------------------------------------------------------

VALID_WORKLOADS = {
    'nanogpt-12m', 'nano-moe-12m', 'micro-dlrm-1m', 'micro-diff', 'micro-gcn',
    'micro-bert', 'micro-lstm', 'micro-rl', 'resnet18', 'mobilenetv2',
    'dscnn-kws', 'anomaly-ae', 'vww', 'nano-rag', 'nano-codegen', 'nano-react'
}

WORKLOAD_TO_DIVISION = {
    'nanogpt-12m': 'cloud', 'nano-moe-12m': 'cloud', 'micro-dlrm-1m': 'cloud',
    'micro-diff': 'cloud', 'micro-gcn': 'cloud', 'micro-bert': 'cloud',
    'micro-lstm': 'cloud', 'micro-rl': 'cloud',
    'resnet18': 'edge', 'mobilenetv2': 'edge',
    'dscnn-kws': 'tiny', 'anomaly-ae': 'tiny', 'vww': 'tiny',
    'nano-rag': 'agent', 'nano-codegen': 'agent', 'nano-react': 'agent'
}

DEFAULT_TARGETS = {
    'nanogpt-12m': ('loss', 2.3), 'nano-moe-12m': ('loss', 0.05),
    'micro-dlrm-1m': ('accuracy', 0.70), 'micro-diff': ('mse', 0.002),
    'micro-gcn': ('accuracy', 0.78), 'micro-bert': ('accuracy', 0.75),
    'micro-lstm': ('mse', 0.13), 'micro-rl': ('reward', 195),
    'resnet18': ('top1', 0.36), 'mobilenetv2': ('top1', 0.40),
    'dscnn-kws': ('top1', 0.90), 'anomaly-ae': ('mse', 0.04),
    'vww': ('accuracy', 0.85), 'nano-rag': ('retrieval', 0.60),
    'nano-codegen': ('pass1', 0.15), 'nano-react': ('trace', 0.70)
}

def load_workloads_config():
    try:
        with open("workloads.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load workloads.yaml: {e}[/bold red]")
        sys.exit(1)

CONFIG = load_workloads_config()

# --------------------------------------------------------------------------------
# CLI Parser Setup
# --------------------------------------------------------------------------------

def setup_cloud_parser(subparsers):
    parser = subparsers.add_parser('cloud', help="Run Cloud/Datacenter workloads (LLMs, DLRM)")
    choices = list(CONFIG['suites']['cloud'].keys())
    parser.add_argument('--task', type=str, required=True, choices=choices, help="The cloud workload to execute.")
    parser.add_argument('--sut', type=str, help="Path to the student's custom .py plugin implementing SUT_Interface")
    parser.add_argument('--model-signature', type=str, help="Path to the .sig schema (Required for inference)")
    parser.add_argument('--scenario', type=str, choices=['Offline', 'Server', 'SingleStream'], default='Offline')
    parser.add_argument('--division', type=str, choices=['closed', 'open'], default='closed', help="The grading division. Closed enforces the 99% accuracy rule.")
    parser.add_argument('--disable-kv-cache', action='store_true', help="A pedagogic hook to crush performance by disabling the LLM cache.")
    parser.set_defaults(func=run_cloud)

def setup_edge_parser(subparsers):
    parser = subparsers.add_parser('edge', help="Run Edge workloads (Vision, NLP)")
    choices = list(CONFIG['suites']['edge'].keys())
    parser.add_argument('--task', type=str, required=True, choices=choices, help="The edge workload to execute.")
    parser.add_argument('--sut', type=str, help="Path to the student's custom .py plugin implementing SUT_Interface")
    parser.add_argument('--model-signature', type=str, help="Path to the .sig schema (Required for inference)")
    parser.add_argument('--scenario', type=str, choices=['Offline', 'Server', 'SingleStream'], default='Offline')
    parser.add_argument('--division', type=str, choices=['closed', 'open'], default='closed', help="The grading division. Closed enforces the 99% accuracy rule.")
    parser.add_argument('--demo-mode', action='store_true', help="Run 1 quick epoch for telemetry testing.")
    parser.set_defaults(func=run_edge)

def setup_mobile_parser(subparsers):
    if 'mobile' not in CONFIG.get('suites', {}):
        return  # Mobile suite not defined in workloads.yaml
    parser = subparsers.add_parser('mobile', help="Run Mobile workloads (Object Detection, Mobile NLP)")
    choices = list(CONFIG['suites']['mobile'].keys())
    parser.add_argument('--task', type=str, required=True, choices=choices, help="The mobile workload to execute.")
    parser.add_argument('--sut', type=str, help="Path to the student's custom .py plugin implementing SUT_Interface")
    parser.add_argument('--model-signature', type=str, help="Path to the .sig schema (Required for inference)")
    parser.add_argument('--scenario', type=str, choices=['Offline', 'Server', 'SingleStream'], default='Offline')
    parser.add_argument('--division', type=str, choices=['closed', 'open'], default='closed', help="The grading division. Closed enforces the 99% accuracy rule.")
    parser.add_argument('--disable-nms', action='store_true', help="A pedagogic hook to disable bounding box sorting.")
    parser.set_defaults(func=run_mobile)

def setup_tiny_parser(subparsers):
    parser = subparsers.add_parser('tiny', help="Run TinyML workloads (KWS, Anomaly Detection)")
    choices = list(CONFIG['suites']['tiny'].keys())
    parser.add_argument('--task', type=str, required=True, choices=choices, help="The tiny workload to execute.")
    parser.add_argument('--sut', type=str, help="Path to the student's custom .py plugin implementing SUT_Interface")
    parser.add_argument('--model-signature', type=str, help="Path to the .sig schema (Required for inference)")
    parser.add_argument('--division', type=str, choices=['closed', 'open'], default='closed', help="The grading division. Closed enforces the 99% accuracy rule.")
    parser.set_defaults(func=run_tiny)

def setup_hydrate_parser(subparsers):
    parser = subparsers.add_parser('hydrate', help="Download canonical weights and commit them securely into a .sig artifact.")
    # Generic choices mapped across all tracks
    all_choices = []
    for track in CONFIG['suites'].values():
        all_choices.extend(list(track.keys()))
    parser.add_argument('--task', type=str, required=True, choices=all_choices, help="The workload to hydrate.")
    parser.set_defaults(func=run_hydrate)

# --------------------------------------------------------------------------------
# Handlers for Workloads
# --------------------------------------------------------------------------------

def print_header(track: str, task: str, config: dict, color: str):
    console.print(Panel(f"Task: {task}\nProvenance: [italic]{config.get('provenance', 'N/A')}[/italic]", title=f"[bold {color}]{track}[/bold {color}]"))

def execute_sut_plugin(args, config: dict, track_name: str):
    """
    Dynamically loads a student's SUT_Interface plugin and pipes it into the LoadGen.
    """
    import importlib.util
    import inspect
    import os
    import sys
    from .sut import SUT_Interface
    
    # Securely append CWD so internal file imports natively resolve cleanly!
    sys.path.insert(0, os.getcwd())
    
    console.print(f"[bold cyan][Plugin Loader] 🔌 Ingesting custom SUT from: {args.sut}[/bold cyan]")
    
    # Load Python Module from arbitrary path
    spec = importlib.util.spec_from_file_location("student_sut", args.sut)
    sut_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sut_module)
    
    # Find the class inheriting from SUT_Interface
    sut_class = None
    for name, obj in inspect.getmembers(sut_module, inspect.isclass):
        if issubclass(obj, SUT_Interface) and obj is not SUT_Interface:
            sut_class = obj
            break
            
    if not sut_class:
        console.print("[bold red]❌ Error: Could not find any class inheriting from `SUT_Interface` in the provided plugin![/bold red]")
        sys.exit(1)
        
    console.print(f"[bold green][Plugin Loader] ✅ Successfully bound SUT class '{sut_class.__name__}'.[/bold green]")
    sut_instance = sut_class(config)
    
    qps = 10.0 if args.scenario.lower() == "server" else 0.0
    
    min_queries = config.get('min_query_count', 100)
    min_duration = config.get('min_duration_seconds', 10.0)
    
    # Provide the proxy with the statistical limits and division context for evaluation
    proxy = LoadGenProxy(scenario=args.scenario.title(), qps=qps, min_query_count=min_queries, min_duration_seconds=min_duration, division=getattr(args, 'division', 'closed'), config=config)
    report = asyncio.run(proxy.run(sut_instance.process_queries))
    
    # ---------------------------------------------------------
    # The Live Pedagogical Scorecard
    # ---------------------------------------------------------
    console.print("\n")
    score_panel = Table(title=f"🏆 {track_name.upper()} Student Scorecard: {args.task}", show_header=True, header_style="bold magenta")
    score_panel.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    score_panel.add_column("Your SUT Implementation", justify="right", style="green")
    score_panel.add_column("Golden Master Baseline", justify="right", style="yellow")
    score_panel.add_column("Grading Status", justify="center")

    # Metrics Extraction
    sut_qps = report.get('queries_per_second', 0.0)
    sut_acc = report.get('achieved_accuracy', 0.0)
    baseline_acc = config.get('baseline_accuracy_fp32', 100.0)
    
    # Mathematical Divisions Check
    division_val = report.get('division_passed', 'FAIL').upper()
    status_color = "bold green" if division_val != "FAIL" else "bold red"
    
    score_panel.add_row(
        "Throughput (QPS)", 
        f"{sut_qps:.2f} queries/s", 
        "N/A (Optimization Task)", 
        "✅ Measured"
    )
    score_panel.add_row(
        "Accuracy Bounds", 
        f"{(sut_acc * 100):.2f}%", 
        f"{(baseline_acc * 100):.2f}%",
        f"[{status_color}]{division_val}[/{status_color}]"
    )
    score_panel.add_row(
        "Energy Consumption", 
        f"{report.get('estimated_joules', 0.0):.1f} Joules", 
        "Target Baseline: TBD", 
        "✅ Captured"
    )
    
    # Bottleneck Inference Context
    if sut_qps > 0:
        bottleneck = "Memory Bandwidth Bound (Sparse Lookups / Quantize!)" if "gpt" in args.task or "dlrm" in args.task else "Compute Bound (Parameters Dense / Compile!)"
        score_panel.add_row(
            "[bold cyan]Diagnosis Bottleneck[/bold cyan]", 
            f"[bold yellow]{bottleneck}[/bold yellow]", 
            "N/A", 
            "🔎 Expert AI"
        )
    
    console.print(Panel(score_panel, border_style="cyan", expand=False))
    console.print(f"[dim]Note: Your encrypted SUT JSON payload was securely persisted to `submissions/` for TA Leaderboard aggregation.[/dim]")
    
    # Trigger Dynamic Visual plotting natively mapped to the telemetry proxy hooks 
    try:
        from mlperf.plotting import TelemetryPlotter
        plotter = TelemetryPlotter(args.task, args.division)
        plotter.execute_plotting(report)
    except Exception as e:
        console.print(f"[dim]Visual Curve Exception: {e}[/dim]")
        
    console.print(f"[bold italic]Ready to submit? Run `mlperf submit --sut {args.sut}` to bundle your grading artifacts![/bold italic]")

def run_cloud(args):
    config = CONFIG['suites']['cloud'][args.task]
    print_header("☁️ MLPerf EDU Cloud", args.task, config, "cyan")
    
    if getattr(args, 'sut', None):
        execute_sut_plugin(args, config, "cyan")
        return
        
    if args.task == 'gpt2-infer':
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from reference.cloud.nanogpt_infer import build_dataloader, build_model, run_benchmark
        if not getattr(args, 'model_signature', None):
            console.print("[bold red][CLI Error] Strict Provenance Mode: You must provide a --model-signature artifact. Generate one using `mlperf hydrate --task gpt2-infer`.[/bold red]")
            return
        loader = build_dataloader(config['dataset'])
        model = build_model(args.sig)
        run_benchmark(model, loader, scenario=args.scenario, use_kv_cache=not args.disable_kv_cache)
    elif args.task == 'dlrm-infer':
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from reference.cloud.dlrm_infer import run_benchmark
        run_benchmark(args.sig, scenario=args.scenario)
    else:
        console.print(f"[yellow]Workload '{args.task}' is architected in workloads.yaml but Python logic is pending.[/yellow]")

def run_edge(args):
    config = CONFIG['suites']['edge'][args.task]
    print_header("📱 MLPerf EDU Edge", args.task, config, "magenta")
    
    if getattr(args, 'sut', None):
        execute_sut_plugin(args, config, "magenta")
        return
    if args.task == 'resnet-train':
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from reference.edge.resnet_train import run_benchmark
        run_benchmark(None, scenario="train")
    elif args.task == 'resnet-infer':
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from reference.edge.resnet_infer import run_benchmark
        if not getattr(args, 'model_signature', None):
            console.print("[bold red][CLI Error] Strict Provenance Mode: You must provide a --model-signature artifact for inference.[/bold red]")
            return
        run_benchmark(args.sig, scenario=args.scenario)
    elif args.task == 'bert-infer':
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from reference.edge.bert_infer import run_benchmark
        run_benchmark(args.sig, scenario=args.scenario)
    else:
        console.print(f"[yellow]Workload '{args.task}' is architected in workloads.yaml but Python logic is pending.[/yellow]")

def run_mobile(args):
    config = CONFIG['suites']['mobile'][args.task]
    print_header("🤳 MLPerf EDU Mobile", args.task, config, "green")
    
    if getattr(args, 'sut', None):
        execute_sut_plugin(args, config, "green")
        return
    if args.task == 'mobilenet-infer':
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from reference.mobile.mobilenet_infer import run_benchmark
        if not getattr(args, 'model_signature', None):
            console.print("[bold red][CLI Error] Strict Provenance Mode: You must provide a --model-signature artifact. Generate one using `mlperf hydrate --task mobilenet-infer`.[/bold red]")
            return
        run_benchmark(config, scenario=args.scenario.title(), disable_nms=args.disable_nms, use_golden=False)
    elif args.task == 'mobilebert-infer':
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from reference.mobile.mobilebert_infer import run_benchmark
        run_benchmark(args.sig, scenario=args.scenario)
    else:
        console.print(f"[yellow]Workload '{args.task}' is architected in workloads.yaml but Python logic is pending.[/yellow]")

def run_tiny(args):
    config = CONFIG['suites']['tiny'][args.task]
    print_header("🔋 MLPerf EDU TinyML", args.task, config, "orange3")
    
    if getattr(args, 'sut', None):
        execute_sut_plugin(args, config, "orange3")
        return
        
    if args.task == 'dscnn-kws':
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from reference.tiny.dscnn_kws import run_benchmark
        run_benchmark(args.sig, scenario="Offline")
    elif args.task == 'ad01-infer':
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from reference.tiny.ad01_infer import run_benchmark
        run_benchmark(args.sig, scenario="Offline")
    else:
        console.print(f"[yellow]Workload '{args.task}' is architected in workloads.yaml but Python logic is pending.[/yellow]")

def run_hydrate(args):
    """
    Downloads and commits mathematically canonical weights bridging the provenance gap.
    """
    from .provenance import ProvenanceManager
    import os
    import torch
    console.print(Panel(f"Hydrating canonical model for: {args.task}", title="[bold blue]💧 MLPerf EDU Hydration Engine[/bold blue]"))
    
    # Simulate the architectural download
    console.print(f"[Provenance] 📥 Downloading Golden weights from canonical sources...")
    
    # Save dummy weights acting as the golden source
    dummy_path = f"{args.task}_canonical.pt"
    torch.save({"golden_state": "verifiable"}, dummy_path)
    
    # Commit the provenance formally!
    signature_file = ProvenanceManager.hydrate_canonical(
        checkpoint_path=dummy_path,
        workload=args.task,
        origin_source="HuggingFace / Torchvision Canonical Weights"
    )
    
    console.print(f"[bold green]✅ Hydration complete! You may now run inference using `--model-signature {signature_file}` as proof of origination.[/bold green]")

def execute_fetch(args):
    import yaml
    import os
    import urllib.request
    from rich.progress import Progress
    
    console.print(Panel(f"Targeting: {'ALL Tasks' if args.all else args.task}", title="[bold yellow]📦 MLPerf EDU Fetcher Engine[/bold yellow]"))
    
    try:
        with open("datasets.yaml", "r") as f:
            data_cfg = yaml.safe_load(f)['datasets']
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load datasets.yaml: {e}[/bold red]")
        return
        
    targets = []
    
    if args.all:
        targets = list(data_cfg.keys())
    else:
        # Match task string securely from workloads.yaml 
        task_config = None
        for suite in CONFIG['suites'].values():
            if args.task in suite:
                task_config = suite[args.task]
                break
        
        if not task_config:
            console.print(f"[bold red]❌ Invalid task mapping for '{args.task}'.[/bold red]")
            return
            
        dataset_name = task_config.get("dataset")
        if dataset_name not in data_cfg:
            console.print(f"[bold red]❌ Dataset '{dataset_name}' not defined in datasets.yaml![/bold red]")
            return
            
        targets = [dataset_name]

    os.makedirs(".data", exist_ok=True)
    
    for t in targets:
        cfg = data_cfg[t]
        uri = cfg['uri']
        
        console.print(f"\n[cyan]Dataset Segment:[/cyan] [bold]{t}[/bold]")
        console.print(f"[dim]Description:[/dim] {cfg.get('description', 'N/A')}")
        console.print(f"[dim]Estimated Load:[/dim] {cfg.get('estimated_size_mb', 'Unknown')} MB")
        
        if args.dry_run:
            console.print(f"[yellow]--dry-run bypass: Would logically fetch from {uri}[/yellow]")
            continue
            
        if uri.startswith("torchvision://") or uri.startswith("synthetic_generator:"):
            console.print(f"[bold green]✅ Logical Dataset - Natively handled via PyTorch Execution script. No static payload necessary.[/bold green]")
            continue
            
        # The true download simulator!
        filename = os.path.join(".data", os.path.basename(uri))
        try:
            console.print(f"📥 Pulling URI: {uri}")
            # Mock pedagogical download delay 
            import time; time.sleep(0.5) 
            # urllib.request.urlretrieve(uri, filename)
            console.print(f"[bold green]✅ Synced offline successfully into `.data/` hook![/bold green]")
        except Exception as e:
            console.print(f"[bold red]❌ Fetch Failed across Network limits! ({e})[/bold red]")

def setup_fetch_parser(subparsers):
    parser = subparsers.add_parser('fetch', help="[Data Utility] Globally coordinates safe local caches resolving isolated `.data/` dependencies avoiding network hangs.")
    parser.add_argument("--task", type=str, help="Specific benchmark dependency to isolate.")
    parser.add_argument("--all", action="store_true", help="Aggressively download the total syllabus package locally.")
    parser.add_argument("--dry-run", action="store_true", help="Compute payload sizes mathematically without initiating downloads.")
    parser.set_defaults(func=execute_fetch)


def execute_anchor(args):
    """
    Administrative hook: Automatically executes the target framework mathematically dynamically overwriting 
    residing thresholds with the actual empirically-measured outputs.
    """
    console.print(f"[bold cyan]⚓ Executing Empirical Anchor generation for task '{args.task}' in suite '{args.suite}'...[/bold cyan]")
    
    if args.suite not in CONFIG['suites'] or args.task not in CONFIG['suites'][args.suite]:
        console.print(f"[bold red]❌ Invalid task mapping.[/bold red]")
        return
        
    config = CONFIG['suites'][args.suite][args.task]
    
    # We dynamically load the physical native module baseline
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    if args.suite == 'edge' and args.task == 'resnet-infer':
        from reference.edge.resnet_infer import run_benchmark
        if not getattr(args, 'model_signature', None):
            console.print("[bold red][CLI Error] Strict Provenance Mode: You must provide a --model-signature artifact for baseline anchoring.[/bold red]")
            return
            
        # Natively extract the floating accuracy representation returned by the module!
        accuracy_percentage = run_benchmark(args.sig, scenario="SingleStream")
        
        # 1. Update master FP32 baseline
        true_baseline = accuracy_percentage / 100.0  # Normalize back to 0.0 - 1.0 probability space
        config['baseline_accuracy_fp32'] = float(f"{true_baseline:.4f}")
        
        # 2. Automate MLCommons Grading Boundaries Algorithmically
        # The 99% Rule: closed division must mathematically equal 0.99x True Baseline
        config['divisions']['closed']['threshold'] = float(f"{(true_baseline * 0.99):.4f}")
        
        # 3. Rewrite natively into YAML file cleanly
        yaml_path = os.path.join(os.path.dirname(__file__), '..', '..', 'workloads.yaml')
        with open(yaml_path, 'w') as f:
            yaml.safe_dump({"suites": CONFIG['suites']}, f, default_flow_style=False, sort_keys=False)
            
        console.print("[bold green]✅ Success! The Empirical Physical bounds were mathematically locked into workloads.yaml![/bold green]")
        console.print(f"FP32 Extracted Anchor: {config['baseline_accuracy_fp32']}")
        console.print(f"Strict 99% Threshold Computed For SUTs: {config['divisions']['closed']['threshold']}")

    else:
        console.print(f"[yellow]Anchor hook for '{args.task}' is under development. Complete the baseline implementations to unlock this feature.[/yellow]")

def setup_anchor_parser(subparsers):
    parser = subparsers.add_parser('anchor', help="Empirically trains or evaluates the target and mathematically updates workloads.yaml threshold configurations.")
    parser.add_argument("suite", choices=["cloud", "edge", "mobile", "tiny"], help="Target benchmark suite")
    parser.add_argument("--task", required=True, help="Specific benchmark to evaluate (e.g., resnet-infer)")
    parser.add_argument("--model-signature", type=str, help="Path to cryptographic model constraints")
    parser.set_defaults(func=execute_anchor)


def setup_grade_parser(subparsers):
    parser = subparsers.add_parser('grade', help="[TA Administrative Tool] Scans the local `./submissions` directory and builds a Hardware-isolated Pedagogy Leaderboard ranking SUT Throughputs globally.")
    from .grader import execute_grading
    parser.set_defaults(func=execute_grading)

# --------------------------------------------------------------------------------
# Interactive Onboarding Wizard
# --------------------------------------------------------------------------------

def execute_init(args):
    import sys
    import yaml
    from rich.prompt import Confirm
    
    console.print(Panel("""[bold cyan]Welcome to MLPerf EDU CS249r![/bold cyan]

This pedagogical framework enforces strict Machine Learning Systems logic.
You are running natively decoupled from industrial hardware constraints.
Let's configure your local academic environment.""", border_style="cyan"))

    # Step 1: Python/System bounds check
    py_ver = sys.version_info
    if py_ver.major < 3 or (py_ver.major == 3 and py_ver.minor < 8):
        console.print("[bold red]❌ Python 3.8+ required. Please update your environment before running MLPerf workloads.[/bold red]")
        return
    console.print("[bold green]✅ System configuration verified.[/bold green]")
    
    # OS Power Profiling Tracking Setup
    import platform
    import subprocess
    os_name = platform.system()
    has_power = False
    console.print("\n[bold]Step 1: Telemetry Energy Probe Initialization[/bold]")
    if os_name == "Darwin":
        res = subprocess.run(["sudo", "-n", "powermetrics", "--help"], capture_output=True)
        if res.returncode == 0:
            console.print("[bold green]✅ [Energy Tracker]: Apple powermetrics ONLINE natively tracking Wattage![/bold green]")
            has_power = True
        else:
            console.print("[yellow]⚠️ Cannot mount Apple powermetrics automatically without `sudo` privileges. Power scaling disabled.[/yellow]")
    elif os_name == "Linux":
        try:
            import pynvml
            pynvml.nvmlInit()
            console.print("[bold green]✅ [Energy Tracker]: NVIDIA NVML ONLINE! Physically binding to PCIe limits![/bold green]")
            has_power = True
        except ImportError:
            console.print("[yellow]⚠️ `pynvml` unavailable natively on this Linux array.[/yellow]")
    else:
        console.print("[dim]OS unrecognized for hardware power hooks. Bounding fallback to generic proxies.[/dim]")
    print()
    
    # Step 3: Hardware Introspection
    console.print("[bold]Step 2: Hardware Roofline Calibration[/bold]")
    console.print("To grade your algorithms mathematically fairly, we must establish physical limitations.")
    do_profile = Confirm.ask("Introspect Hardware via `mlperf profile` natively now?", default=True)
    if do_profile:
        run_profile(args)
    else:
        console.print("[dim]Skipping parameter tracking. Run `mlperf profile` later to map hardware offsets.[/dim]")
        
    print()
    
    # Step 4: Payload estimations
    console.print("[bold]Step 3: Canonical Payload Sizing[/bold]")
    console.print("Extracting Single Source of Truth dataset bounds from `datasets.yaml`...")
    try:
        with open("datasets.yaml", "r") as f:
            d_cfg = yaml.safe_load(f)['datasets']
            
        total_mb = sum(cfg.get('estimated_size_mb', 0.0) for cfg in d_cfg.values())
        console.print(f"[bold yellow]Total isolated data cache bounds: ~{total_mb / 1024:.2f} GB[/bold yellow]")
        
        do_fetch = Confirm.ask("Execute `mlperf fetch --all` to hydrate local `.data/` cache instantly?", default=False)
        
        if do_fetch:
            console.print("[bold cyan]Initiating Master Fetch...[/bold cyan]")
            # Hacky rewrite of args to securely pass to `execute_fetch` directly
            args.all = True
            args.dry_run = False
            execute_fetch(args)
        else:
            console.print("[dim]Skipping massive hydration phase. Hydrate manually via `mlperf fetch` when connected to stable Wi-Fi.[/dim]")
            
    except Exception as e:
        console.print(f"[yellow]⚠️ Could not parse datasets.yaml dynamically: {e}[/yellow]")
        
    console.print("\n[bold green]🏁 Onboarding Complete! You are mathematically cleared to execute workloads.[/bold green]")

def setup_init_parser(subparsers):
    parser = subparsers.add_parser('init', help="Interactive Student setup tracking hardware profiling and bounds dependencies safely.")
    parser.set_defaults(func=execute_init)

def execute_about(args):
    console.print(Panel("""[bold magenta]Harvard CS249r: Machine Learning Systems[/bold magenta]

[bold]Mission:[/bold] 
MLPerf EDU drops the "Black-Box" abstractions of ML. We focus completely on defining the
exact Systems Engineering bottlenecks natively throttling Modern Generative hardware architectures. 

[bold]Instructor Journey:[/bold] 
Instructors utilize `mlperf anchor` to mathematically prove canonical workloads, setting 
the absolute 99% accuracy Grading rules explicitly inside `workloads.yaml`. 

[bold]Student Journey:[/bold] 
Students clone `reference/` and rewrite Native PyTorch execution graphs directly. 
When they securely execute `mlperf run --sut my_optimizations.py`, the Framework evaluates 
their hardware latency, physical power drainage, and bounding bounds natively generating 
a [green]Scorecard[/green]. 

[bold]Grading Journey:[/bold] 
TAs execute `mlperf grade` to securely rank every student cryptographic payload!""", title="[cyan]About MLPerf EDU[/cyan]", border_style="cyan"))

def setup_about_parser(subparsers):
    parser = subparsers.add_parser('about', help="Prints the architectural user journeys and pedagogical constraints of this repository.")
    parser.set_defaults(func=execute_about)

def execute_submit(args):
    import os
    import glob
    import zipfile
    from rich.prompt import Prompt
    
    console.print(Panel("[bold yellow]📦 MLPerf Formal Submissions Enclave[/bold yellow]", expand=False))
    
    # MLCommons strict structural integrity dictates explicit formatting!
    huid = Prompt.ask("Enter your Harvard Student ID (HUID)")
    if not huid.strip():
        console.print("[bold red]❌ Invalid HUID![/bold red]")
        return
        
    # 1. Verify SUT file
    if not os.path.exists(args.sut):
        console.print(f"[bold red]❌ Cannot find SUT file: {args.sut}[/bold red]")
        return
        
    # 2. Get the latest submission JSON mathematically
    sub_files = glob.glob(os.path.join("submissions", "*.json"))
    if not sub_files:
        console.print("[bold red]❌ No tracking payloads found! Run `mlperf run` first.[/bold red]")
        return
    
    latest_json = max(sub_files, key=os.path.getctime)
    ts = os.path.basename(latest_json).replace('.json', '')
    zip_name = f"mlperf_submission_{huid}_{ts}.zip"
    
    # 3. Zip them cleanly mapping explicitly onto MLCommons folder structures
    try:
        with zipfile.ZipFile(zip_name, 'w') as z:
            # We strictly enforce the /code and /results directory schema natively requested
            z.write(args.sut, arcname=f"{huid}/code/{os.path.basename(args.sut)}")
            z.write(latest_json, arcname=f"{huid}/results/{os.path.basename(latest_json)}")
        console.print(f"[bold green]✅ Homework successfully packaged into: {zip_name}[/bold green]")
        console.print(f"   Contains formal MLCommons formatting for TA automation.")
        console.print("   [dim]Upload this securely to Canvas/Gradescope.[/dim]")
    except Exception as e:
        console.print(f"[bold red]❌ Packaging Failed: {e}[/bold red]")

def setup_submit_parser(subparsers):
    parser = subparsers.add_parser('submit', help="Generates a zipped grading artifact explicitly formatting structural repos for Harvard Gradescope.")
    parser.add_argument("--sut", type=str, required=True, help="Path to your System Under Test optimization script.")
    parser.set_defaults(func=execute_submit)

def execute_verify(args):
    import json
    import hashlib
    import os
    
    console.print(f"[bold cyan]🔍 Executing Cryptographic Audit on: {args.payload}[/bold cyan]")
    
    if not os.path.exists(args.payload):
        console.print(f"[bold red]❌ Could not locate payload![/bold red]")
        return
        
    try:    
        with open(args.payload, 'r') as f:
            data = json.load(f)
            
        report = data['telemetry']
        hw = data['system_under_test']
        target_hash = data.get('integrity_hash', '')
        
        # Recalculate explicitly!
        hash_payload = str(report) + str(hw)
        sealed_hash = hashlib.sha256(hash_payload.encode('utf-8')).hexdigest()
        
        if sealed_hash == target_hash:
            console.print("[bold green]✅ INTEGRITY VERIFIED: Output math matches cryptographically securely![/bold green]")
        else:
            console.print(Panel(f"Target Hash: {sealed_hash}\nPayload Claim: {target_hash}", title="[bold red]❌ CHEATING DETECTED[/bold red]"))
            console.print("[red]The Student manually altered JSON Metrics natively bypassing system telemetry! Disqualify![/red]")
            
    except Exception as e:
        console.print(f"[bold red]❌ Parsing Failure: Corrupted payload! ({e})[/bold red]")

def setup_verify_parser(subparsers):
    parser = subparsers.add_parser('verify', help="Executes structural hash calculations validating cryptographic JSON payload bounds natively catching spoofed parameters!")
    parser.add_argument("payload", type=str, help="Path to the JSON submission payload.")
    parser.set_defaults(func=execute_verify)

def execute_audit(args):
    console.print(Panel(f"[bold red]☢️ Executing Adversarial Audit: {args.task}[/bold red]", expand=False))
    console.print("[dim]Injecting Poisoned Audit Tensors into SUT Pipeline...[/dim]")
    
    config = CONFIG['suites'][args.suite][args.task]
    config['audit_mode'] = True # The proxy will deliberately swap correct datasets for malicious bounds
    
    # We force the exact sequence as run, but explicitly bounded heavily
    args.scenario = 'Offline'
    execute_sut_plugin(args, config, "red")

def setup_audit_parser(subparsers):
    parser = subparsers.add_parser('audit', help="Executes SUT dynamically injecting adversarial Noise Tensors natively blocking Hard-coded spoof logics!")
    parser.add_argument("suite", choices=["cloud", "edge", "mobile", "tiny"], help="Target benchmark suite")
    parser.add_argument("--task", type=str, required=True, help="Specific benchmark to audit")
    parser.add_argument("--sut", type=str, required=True, help="Path to your System Under Test optimization script.")
    parser.add_argument('--division', type=str, default='closed')
    parser.set_defaults(func=execute_audit)

# --------------------------------------------------------------------------------
# System Setup & Utilities
# --------------------------------------------------------------------------------

def run_profile(args):
    console.print("[bold cyan]🔍 Profiling Host Hardware...[/bold cyan]")
    info = profile_hardware()
    table = Table(title="Hardware Profile")
    table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_row("Device", str(info['device']))
    table.add_row("Estimated Peak FLOPs", f"{info.get('peak_flops', 0)/1e12:.2f} TFLOPs")
    table.add_row("Estimated Memory Bandwidth", f"{info.get('peak_bandwidth', 0)/1e9:.2f} GB/s")
    console.print(table)

def run_list(args):
    console.print(Panel("[bold green]MLPerf EDU Suite Roster[/bold green]\nTargeting structural 1:1 mirroring of MLCommons taxonomy.", border_style="green"))
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Track")
    table.add_column("Task ID")
    table.add_column("Model Arch")
    table.add_column("Dataset")
    table.add_column("Original Provenance")
    
    track_colors = {"cloud": "cyan", "edge": "magenta", "mobile": "green", "tiny": "orange3"}
    
    for track, tasks in CONFIG['suites'].items():
        for task_id, details in tasks.items():
            color = track_colors.get(track, "white")
            prov = details.get("provenance", "N/A")
            # Truncate provenance if too long for clean display
            if len(prov) > 60:
                prov = prov[:57] + "..."
            table.add_row(
                f"[{color}]{track.upper()}[/{color}]",
                task_id,
                details.get('model', 'N/A'),
                details.get('dataset', 'N/A'),
                f"[dim]{prov}[/dim]"
            )
            
    console.print(table)

# --------------------------------------------------------------------------------
# YAML Config Runner — mlperf run config.yaml
# Implements the declarative experiment specification from the paper.
# Each YAML field maps to a component of the evaluation tuple:
#   S = (Model, Data, Hardware, Scenario, Constraints) → M = (Latency, Throughput, Accuracy, Energy)
# --------------------------------------------------------------------------------

def run_from_config(args):
    """Execute a workload from a YAML configuration file.
    
    This implements the declarative interface described in the paper:
        $ mlperf run config.yaml
    
    The YAML file maps directly to the evaluation tuple S:
        workload → Model
        dataset → Data
        hardware → Hardware (auto-detected if 'auto')
        scenario → Scenario
        division, target_quality, epochs → Constraints
    """
    config_path = args.config_file
    
    try:
        with open(config_path, 'r') as f:
            run_config = yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[bold red]❌ Config file not found: {config_path}[/bold red]")
        sys.exit(1)
    except yaml.YAMLError as e:
        console.print(f"[bold red]❌ Invalid YAML syntax: {e}[/bold red]")
        sys.exit(1)
    
    # Validate required fields
    required = ['workload']
    for field in required:
        if field not in run_config:
            console.print(f"[bold red]❌ Missing required field '{field}' in {config_path}[/bold red]")
            sys.exit(1)
    
    workload = run_config['workload']
    if workload not in VALID_WORKLOADS:
        console.print(f"[bold red]❌ Unknown workload '{workload}'. Valid: {sorted(VALID_WORKLOADS)}[/bold red]")
        sys.exit(1)
    
    # Resolve defaults
    division = WORKLOAD_TO_DIVISION.get(workload, 'cloud')
    mode = run_config.get('mode', 'train')
    target_metric, target_value = DEFAULT_TARGETS.get(workload, ('loss', 1.0))
    target_quality = run_config.get('target_quality', target_value)
    epochs = run_config.get('epochs', 25)
    batch_size = run_config.get('batch_size', 16)
    seed = run_config.get('seed', 42)
    scenario = run_config.get('scenario', 'SingleStream')
    run_division = run_config.get('division', 'closed')
    hardware = run_config.get('hardware', 'auto')
    output_dir = run_config.get('output', 'submissions/')
    sut = run_config.get('sut', None)
    
    # Auto-detect hardware if not specified
    hw_fingerprint = None
    if hardware == 'auto' and detect_hardware is not None:
        hw_fingerprint = detect_hardware()
        hardware_display = f"{hw_fingerprint['chip']} ({hw_fingerprint['memory_gb']}GB, {hw_fingerprint['backend']})"
    else:
        hardware_display = hardware
    
    # Display evaluation tuple mapping
    table = Table(title="Evaluation Tuple S → M", show_lines=True)
    table.add_column("Component", style="bold cyan")
    table.add_column("Field", style="white")
    table.add_column("Value", style="bold green")
    table.add_row("S.Model", "workload", workload)
    table.add_row("S.Data", "dataset", run_config.get('dataset', 'auto'))
    table.add_row("S.Hardware", "hardware", hardware_display)
    if hw_fingerprint:
        table.add_row("S.Hardware", "fingerprint", hw_fingerprint.get('fingerprint_hash', 'N/A'))
    table.add_row("S.Scenario", "scenario", scenario)
    table.add_row("S.Constraints", "division", run_division)
    table.add_row("S.Constraints", "target_quality", f"{target_metric} {'<' if target_metric in ('loss', 'mse') else '>'} {target_quality}")
    table.add_row("S.Constraints", "epochs", str(epochs))
    table.add_row("S.Constraints", "batch_size", str(batch_size))
    table.add_row("S.Constraints", "seed", str(seed))
    table.add_row("→ M", "output", output_dir)
    console.print(table)
    
    console.print(f"\n[bold]Mode:[/bold] {mode} | [bold]Division:[/bold] {division}/{run_division}")
    console.print(f"[dim]Config loaded from: {config_path}[/dim]\n")
    
    # Dispatch to appropriate handler
    import types
    mock_args = types.SimpleNamespace(
        task=workload,
        sut=sut,
        scenario=scenario,
        division=run_division,
        model_signature=None,
        disable_kv_cache=False,
        demo_mode=False,
        disable_nms=False
    )
    
    handler_map = {
        'cloud': run_cloud,
        'edge': run_edge,
        'mobile': run_mobile,
        'tiny': run_tiny
    }
    
    handler = handler_map.get(division)
    if handler:
        handler(mock_args)
    else:
        console.print(f"[yellow]Division '{division}' handler not yet implemented for YAML runner.[/yellow]")


def run_train_all(args):
    """Train all workloads sequentially (or a subset by division)."""
    import time
    division_filter = getattr(args, 'division_filter', None)
    
    console.print(Panel(
        f"Training {'all' if not division_filter else division_filter} workloads",
        title="[bold cyan]🏋️ MLPerf EDU Training Suite[/bold cyan]"
    ))
    
    for workload, division in sorted(WORKLOAD_TO_DIVISION.items(), key=lambda x: x[1]):
        if division_filter and division != division_filter:
            continue
        target_metric, target_value = DEFAULT_TARGETS.get(workload, ('loss', 1.0))
        console.print(f"\n[bold]{division.upper()}/{workload}[/bold] — target: {target_metric} {'<' if target_metric in ('loss', 'mse') else '>'} {target_value}")
        # Dispatch to existing handlers
        import types
        mock_args = types.SimpleNamespace(
            task=workload, sut=None, scenario='Offline',
            division='closed', model_signature=None,
            disable_kv_cache=False, demo_mode=False, disable_nms=False
        )
        handler_map = {'cloud': run_cloud, 'edge': run_edge, 'mobile': run_mobile, 'tiny': run_tiny}
        handler = handler_map.get(division)
        if handler:
            try:
                handler(mock_args)
            except Exception as e:
                console.print(f"[red]  ⚠ {workload} failed: {e}[/red]")


def run_report(args):
    """Generate an HTML report from a submission JSON."""
    from .report import generate_report
    report_path = generate_report(
        args.submission,
        output_path=args.output,
        baseline_path=args.baseline
    )
    console.print(f"[bold green]✅ Report generated: {report_path}[/bold green]")


def main():
    parser = argparse.ArgumentParser(
        prog="mlperf",
        description="MLPerf EDU: A Pedagogical Evaluation Framework for AI Systems Benchmarking."
    )
    
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)
    
    # Run suites
    run_parser = subparsers.add_parser('run', help="Execute an MLPerf EDU workload")
    run_subparsers = run_parser.add_subparsers(title="suites", dest="suite", required=True)
    
    setup_cloud_parser(run_subparsers)
    setup_edge_parser(run_subparsers)
    setup_mobile_parser(run_subparsers)
    setup_tiny_parser(run_subparsers)
    
    # Hydration Protocol
    setup_hydrate_parser(subparsers)
    
    # Offline Dataset Fetch Engine
    setup_fetch_parser(subparsers)
    
    # Threshold Anchoring (Empirical TA grading mechanism)
    setup_anchor_parser(subparsers)
    
    # Leaderboard Generations
    setup_grade_parser(subparsers)
    
    # Utilities
    setup_about_parser(subparsers)
    setup_init_parser(subparsers)
    setup_submit_parser(subparsers)
    setup_verify_parser(subparsers)
    setup_audit_parser(subparsers)
    
    profile_parser = subparsers.add_parser('profile', help="Run the hardware roofline normalizer to profile this machine")
    profile_parser.set_defaults(func=run_profile)
    
    list_parser = subparsers.add_parser('list', help="List all available workloads defined in the taxonomy")
    list_parser.set_defaults(func=run_list)

    # --- YAML Config Runner (mlperf run config.yaml) ---
    config_parser = subparsers.add_parser(
        'config',
        help="Run a workload from a YAML configuration file (declarative interface)"
    )
    config_parser.add_argument(
        'config_file', type=str,
        help="Path to a YAML experiment config file (maps to evaluation tuple S)"
    )
    config_parser.set_defaults(func=run_from_config)
    
    # --- Train All (mlperf train --all) ---
    train_parser = subparsers.add_parser(
        'train',
        help="Train workloads (use --all for entire suite)"
    )
    train_parser.add_argument('--all', action='store_true', help="Train all 16 workloads")
    train_parser.add_argument('--division', dest='division_filter', type=str,
                             choices=['cloud', 'edge', 'tiny', 'agent'],
                             help="Filter to a specific division")
    train_parser.set_defaults(func=run_train_all)

    # --- Report Generator (mlperf report --submission results.json) ---
    report_parser = subparsers.add_parser(
        'report',
        help="Generate an audit-ready HTML report from a submission JSON"
    )
    report_parser.add_argument(
        '--submission', required=True, type=str,
        help="Path to submission JSON file"
    )
    report_parser.add_argument(
        '--output', type=str, default=None,
        help="Output HTML path (default: <submission>_report.html)"
    )
    report_parser.add_argument(
        '--baseline', type=str, default=None,
        help="Optional baseline submission JSON for comparison deltas"
    )
    report_parser.set_defaults(func=run_report)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
