# generate_appendix.py
"""
mlsysim Appendix Generator
==========================
Generates Quarto-compatible Markdown tables for the textbook's backmatter.
Extracts live data from the mlsysim Hardware and Model registries.
"""

from mlsysim.core.constants import Q_
from mlsysim.hardware.registry import Hardware
from mlsysim.models.registry import Models

def fmt_q(q: Q_, precision: int = 1) -> str:
    """Format a quantity for the table."""
    if q is None: return "---"
    return f"{q.magnitude:,.{precision}f} {q.units:~P}"

def generate_hardware_appendix():
    """Generates the Hardware Specifications table for the appendix."""
    header = "| Accelerator | Year | Peak FP16 | Memory BW | Memory Capacity | TDP |\n"
    divider = "|:---|:---:|:---:|:---:|:---:|:---:|\n"
    
    rows = []
    # Cloud Tiers
    for h in [Hardware.A100, Hardware.H100, Hardware.H200, Hardware.MI300X]:
        row = f"| {h.name} | {h.release_year} | {fmt_q(h.compute.peak_flops)} | {fmt_q(h.memory.bandwidth)} | {fmt_q(h.memory.capacity)} | {fmt_q(h.tdp, 0)} |"
        rows.append(row)
        
    return header + divider + "\n".join(rows)

def generate_model_appendix():
    """Generates the Model Workload table for the appendix."""
    header = "| Model | Architecture | Parameters | Inference FLOPS | Layers |\n"
    divider = "|:---|:---:|:---:|:---:|:---:|\n"
    
    rows = []
    for m in [Models.GPT2, Models.GPT3, Models.ResNet50, Models.MobileNetV2]:
        row = f"| {m.name} | {m.architecture} | {fmt_q(m.parameters)} | {fmt_q(m.inference_flops)} | {m.layers or '---'} |"
        rows.append(row)
        
    return header + divider + "\n".join(rows)

if __name__ == "__main__":
    print("## Hardware Specifications Appendix\n")
    print(generate_hardware_appendix())
    print("\n\n## Model Workload Appendix\n")
    print(generate_model_appendix())
