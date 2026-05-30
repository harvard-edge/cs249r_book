import typer
import platform
import subprocess
import shutil
from typing import Optional
from mlsysim.cli.context import OUTPUT_FORMAT_HELP, resolve_output_format
from mlsysim.cli.exceptions import error_shield
from mlsysim.cli.schemas import _resolve_model
from mlsysim.cli.renderers import print_json
from mlsysim.show import info, banner
from mlsysim.hardware import Hardware
from mlsysim.engine.evaluation import SystemEvaluator

def _get_gpu_info():
    """Attempt to detect local GPU specifications."""
    gpu_name = "Generic Accelerator"
    if platform.system() == "Darwin":
        try:
            cpu_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            if "Apple" in cpu_info:
                return f"Apple {cpu_info.split(' ')[1]} Series", "MacBook"
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            pass
    elif platform.system() == "Linux" or platform.system() == "Windows":
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode().strip()
                return out, "NVIDIA GPU"
            except (subprocess.CalledProcessError, FileNotFoundError, OSError):
                pass
    return gpu_name, "CPU/Generic"

def audit_main(
    ctx: typer.Context,
    workload: str = typer.Option("Llama3_8B", "--workload", "-w", help="Workload to audit against (e.g. Llama3_8B, ResNet50)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help=OUTPUT_FORMAT_HELP),
):
    """
    **[Audit] Profile local hardware against the Iron Law.**

    Detects your local CPU/GPU and benchmarks its theoretical capability against the textbook's hardware registry.
    """
    output_format = resolve_output_format(ctx, output, supported={"text", "json", "markdown"})

    with error_shield(output_format=output_format):
        # 1. Hardware Detection
        gpu_name, gpu_type = _get_gpu_info()
        os_info = f"{platform.system()} {platform.release()}"

        # 2. Mapping to Registry
        # For the audit, we'll pick a "reference" if we can't find the exact match
        if "H100" in gpu_name:
            ref_hardware = Hardware.Cloud.H100
        elif "A100" in gpu_name:
            ref_hardware = Hardware.Cloud.A100
        elif "Apple" in gpu_name:
            ref_hardware = Hardware.Workstation.MacBookM3Max
        else:
            ref_hardware = Hardware.Edge.JetsonOrinNX

        # 3. Running Simulation
        model_obj = _resolve_model(workload)
        benchmark_precision = "fp16"

        eval_obj = SystemEvaluator.evaluate(
            scenario_name=f"Audit: {gpu_name} running {workload}",
            model_obj=model_obj,
            hardware_obj=ref_hardware,
            batch_size=1,
            precision=benchmark_precision,
            efficiency=0.45
        )

        # 4. H100 comparison baseline
        h100_eval = SystemEvaluator.evaluate(
            scenario_name="H100 Baseline",
            model_obj=model_obj,
            hardware_obj=Hardware.Cloud.H100,
            batch_size=1,
            precision=benchmark_precision,
            efficiency=0.45
        )

        ratio = h100_eval.performance.metrics['latency'] / eval_obj.performance.metrics['latency']

        if output_format == "json":
            payload = {
                "workload": workload,
                "environment": {
                    "os": os_info,
                    "processor": platform.processor(),
                    "accelerator": gpu_name,
                    "type": gpu_type,
                },
                "reference_hardware": ref_hardware.name,
                "estimated_latency_ms": float(eval_obj.performance.metrics["latency"]),
                "estimated_throughput": float(eval_obj.performance.metrics["throughput"]),
                "bottleneck": eval_obj.performance.summary,
                "h100_relative_speed": float(ratio),
                "benchmark": f"local estimate vs H100 estimate at {benchmark_precision}",
            }
            print_json(payload)
            return

        if output_format == "markdown":
            print(f"## MLSys·im Local Hardware Audit: {workload}")
            print("\n### Local Environment")
            print(f"- **OS**: {os_info}")
            print(f"- **Processor**: {platform.processor()}")
            print(f"- **Accelerator**: {gpu_name}")
            print(f"- **Type**: {gpu_type}")
            print("\n### Theoretical Results")
            print(f"- **Reference match**: {ref_hardware.name}")
            print(f"- **Estimated latency**: {eval_obj.performance.metrics['latency']:.2f} ms")
            print(f"- **Estimated throughput**: {eval_obj.performance.metrics['throughput']:.2f} tokens/s")
            print(f"- **Bottleneck**: {eval_obj.performance.summary}")
            print(f"- **H100-relative speed**: {ratio:.2f}x")
            return

        banner("MLSys·im LOCAL HARDWARE AUDIT")

        info("Local Environment",
             OS=os_info,
             Processor=platform.processor(),
             Accelerator=gpu_name,
             Type=gpu_type)

        info("Audit Results (Theoretical)",
             Workload=workload,
             Reference_Match=ref_hardware.name,
             Est_Latency=f"{eval_obj.performance.metrics['latency']:.2f} ms",
             Est_Throughput=f"{eval_obj.performance.metrics['throughput']:.2f} tokens/s",
             Bottleneck=eval_obj.performance.summary)

        print("\n" + "-" * 44)
        print(f"H100-relative speed: {ratio:.2f}x")
        print(f"Benchmark: local estimate vs H100 estimate at {benchmark_precision}")
        print("-" * 44)
