import argparse
import sys
import json
from .core.config import SimulationConfig, load_config
from .core.scenarios import Scenario
from .core.engine import Engine
from .models.registry import Models
from .hardware.registry import Hardware
from .systems.registry import Systems

def print_scorecard(config_data):
    try:
        config = load_config(config_data)
        
        m_obj = getattr(Models, config.model, None)
        h_obj = getattr(Hardware, config.hardware, None)
        
        if not m_obj:
            print(f"Error: Model '{config.model}' not found in registry.")
            sys.exit(1)
        if not h_obj:
            print(f"Error: Hardware '{config.hardware}' not found in registry.")
            sys.exit(1)

        # Basic Single Node Evaluation
        print(f"--- MLSysim Single Node Evaluation ---")
        print(f"Model: {config.model}")
        print(f"Hardware: {config.hardware}")
        print(f"Batch Size: {config.batch_size}")
        print("-" * 40)
        
        perf = Engine.solve(m_obj, h_obj, batch_size=config.batch_size, precision=config.precision, efficiency=config.efficiency)
        print(perf.summary())
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="MLSysim CLI: First-Principles Infrastructure Modeling")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a hardware/model configuration")
    eval_parser.add_argument("--model", required=True, help="Model name from registry (e.g., Llama2_70B)")
    eval_parser.add_argument("--hardware", required=True, help="Hardware name from registry (e.g., H100)")
    eval_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    eval_parser.add_argument("--precision", type=str, default="fp16", help="Precision (e.g., fp16)")
    eval_parser.add_argument("--efficiency", type=float, default=0.5, help="Compute efficiency (MFU)")
    
    args = parser.parse_args()
    
    if args.command == "evaluate":
        config_data = {
            "model": args.model,
            "hardware": args.hardware,
            "batch_size": args.batch_size,
            "precision": args.precision,
            "efficiency": args.efficiency
        }
        print_scorecard(config_data)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
