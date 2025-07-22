#!/usr/bin/env python3
"""
Automated LLM Model Optimization Experiments

This script runs comprehensive experiments to find the optimal Ollama model
and explanation length for cross-reference generation. It's designed to run
unattended while you're away from the computer.

Usage:
    python3 run_experiments.py

Results will be saved in scripts/llm_experiments/results/
"""

import sys
import time
import traceback
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from experiment_runner import ExperimentRunner

def main():
    """Run the complete experiment suite with error handling"""
    
    print("🚀 Starting Automated LLM Optimization Experiments")
    print("=" * 60)
    print("This will run comprehensive experiments to find:")
    print("  1. The best Ollama model for cross-reference explanations")
    print("  2. The optimal explanation length")
    print("  3. Data-driven recommendations")
    print()
    print("⏱️  Expected duration: 30-60 minutes depending on available models")
    print("📁 Results will be saved to scripts/llm_experiments/results/")
    print("=" * 60)
    
    try:
        # Initialize experiment runner
        runner = ExperimentRunner()
        
        # Check if Ollama is running
        available_models = runner.check_available_models()
        if not available_models:
            print("❌ ERROR: No Ollama models available!")
            print("Please ensure:")
            print("  1. Ollama is installed and running")
            print("  2. At least one model is pulled (e.g., ollama pull qwen2.5:7b)")
            print("  3. Ollama is accessible at http://localhost:11434")
            return 1
        
        print(f"✅ Found {len(available_models)} models - starting experiments...\n")
        
        # Run the full experiment suite
        start_time = time.time()
        results = runner.run_full_experiment_suite()
        end_time = time.time()
        
        duration = end_time - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        print(f"\n🎉 EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total duration: {minutes}m {seconds}s")
        
        # Show quick summary of recommendations
        if "recommendations" in results:
            recs = results["recommendations"]
            if "recommendations" in recs:
                print(f"\n🎯 QUICK RECOMMENDATIONS:")
                if "model" in recs["recommendations"]:
                    model_rec = recs["recommendations"]["model"]
                    print(f"   📦 Best Model: {model_rec['recommended']}")
                if "length" in recs["recommendations"]:
                    length_rec = recs["recommendations"]["length"]
                    print(f"   📏 Best Length: {length_rec['recommended']}")
        
        print(f"\n📁 Detailed results saved in: scripts/llm_experiments/results/")
        print("   Check the recommendations_latest.json file for full analysis!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Experiments interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nPlease check:")
        print("  1. Ollama is running and accessible")
        print("  2. Required Python packages are installed")
        print("  3. Network connectivity is stable")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 