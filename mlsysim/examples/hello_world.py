"""
Hello World: Ten Minutes to mlsysim
===================================
This tutorial demonstrates the end-to-end workflow of mlsysim:
1. Load a Model and Hardware.
2. Solve single-node performance.
3. Scale to a fleet.
4. Calculate Sustainability and Economics.
"""

import mlsysim
from mlsysim import load_config

def main():
    print("--- 1. Define Your Simulation ---")
    user_choice = {
        "model": "ResNet50",
        "hardware": "A100",
        "batch_size": 32,
        "fleet_size": 128,
        "region": "Quebec"
    }
    
    # load_config automatically validates physical feasibility!
    config = load_config(user_choice)
    print("Config Validated: " + config.model + " on " + config.hardware + " in " + config.region + "\n")

    print("--- 2. Single-Node Performance (The Iron Law) ---")
    model = getattr(mlsysim.Models, config.model)
    hardware = getattr(mlsysim.Hardware, config.hardware)
    
    perf = mlsysim.Engine.solve(model, hardware, batch_size=config.batch_size)
    print("Latency:    " + str(perf.latency))
    print("Throughput: " + str(perf.throughput))
    print("Bottleneck: " + perf.bottleneck + "\n")

    print("--- 3. Scenario Evaluation & Visualization ---")
    # Using a vetted lighthouse scenario
    scenario = mlsysim.Applications.AutoDrive
    evaluation = scenario.evaluate()
    print(evaluation.scorecard())
    
    # Visual Scorecard
    fig, ax = mlsysim.plot_evaluation_scorecard(evaluation)
    print("\nVisual Scorecard generated.")

    print("\nSimulation Complete. Check mlsysbook.ai for advanced labs!")

if __name__ == "__main__":
    main()
