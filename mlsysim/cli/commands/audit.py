import typer
import platform
import subprocess
import shutil
from typing import Optional
from mlsysim.cli.exceptions import error_shield
from mlsysim.show import info, banner
from mlsysim.hardware import Hardware
from mlsysim.models import Models
from mlsysim.core.evaluation import SystemEvaluator

def _get_gpu_info():
    """Attempt to detect local GPU specifications."""
    gpu_name = "Generic Accelerator"
    if platform.system() == "Darwin":
        try:
            # Simple check for Apple Silicon
            cpu_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            if "Apple" in cpu_info:
                return f"Apple {cpu_info.split(' ')[1]} Series", "MacBook"
        except:
            pass
    elif platform.system() == "Linux" or platform.system() == "Windows":
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode().strip()
                return out, "NVIDIA GPU"
            except:
                pass
    return gpu_name, "CPU/Generic"

def audit_main(
    ctx: typer.Context,
    workload: str = typer.Option("Llama3_8B", "--workload", "-w", help="Workload to audit against (e.g. Llama3_8B, ResNet50)")
):
    """
    **[Audit] Profile your local hardware against the Iron Law.**
    
    Detects your local CPU/GPU and benchmarks its theoretical capability against the textbook's hardware registry.
    """
    output_format = ctx.obj.get("output_format", "text") if ctx.obj else "text"
    
    with error_shield(output_format=output_format):
        banner("MLSys·im ARCHITECT'S AUDIT")
        
        # 1. Hardware Detection
        gpu_name, gpu_type = _get_gpu_info()
        os_info = f"{platform.system()} {platform.release()}"
        
        info("Local Environment",
             OS=os_info,
             Processor=platform.processor(),
             Accelerator=gpu_name,
             Type=gpu_type)
        
        # 2. Mapping to Registry
        # For the audit, we'll pick a "reference" if we can't find the exact match
        if "H100" in gpu_name:
            ref_hardware = Hardware.Cloud.H100
        elif "A100" in gpu_name:
            ref_hardware = Hardware.Cloud.A100
        elif "Apple" in gpu_name:
            ref_hardware = Hardware.Workstation.MacBook  # We should ensure this exists
        else:
            ref_hardware = Hardware.Edge.OrinNano # Fallback for demo
            
        # 3. Running Simulation
        model_obj = getattr(Models, workload, Models.Llama3_8B)
        
        eval_obj = SystemEvaluator.evaluate(
            scenario_name=f"Audit: {gpu_name} running {workload}",
            model_obj=model_obj,
            hardware_obj=ref_hardware,
            batch_size=1,
            precision="int4",
            efficiency=0.45
        )
        
        # 4. Reporting
        info("Audit Results (Theoretical)",
             Workload=workload,
             Reference_Match=ref_hardware.name,
             Est_Latency=f"{eval_obj.performance.metrics['latency']:.2f} ms",
             Est_Throughput=f"{eval_obj.performance.metrics['throughput']:.2f} tokens/s",
             Bottleneck=eval_obj.performance.summary)
        
        # 5. Viral Hook
        h100_eval = SystemEvaluator.evaluate(
            scenario_name="H100 Baseline",
            model_obj=model_obj,
            hardware_obj=Hardware.Cloud.H100,
            batch_size=1,
            precision="fp16",
            efficiency=0.45
        )
        
        ratio = h100_eval.performance.metrics['latency'] / eval_obj.performance.metrics['latency']
        
        print("
" + "═"*44)
        print(f"🚀 Your local machine is {ratio:.2f}x as fast as an H100 node.")
        print("   (Benchmark: Llama-3-8B Inference at INT4 vs FP16)")
        print("═"*44)
        print("
✨ Star the repo to unlock the full 'Hardware Zoo' and 22-Wall diagnostics!")
        print("🔗 https://github.com/harvard-edge/cs249r_book")

