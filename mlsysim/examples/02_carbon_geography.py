#!/usr/bin/env python3
"""
Example 2: The Carbon Impact of Geography
-----------------------------------------
This script demonstrates the "Macro" capabilities of the framework,
showing how the physical location of a datacenter radically alters 
the environmental impact of training a model.
"""
import mlsysim

def main():
    print("Evaluating a 30-day training run on 256x H100 GPUs...\n")

    # 1. Define the Cluster
    fleet = mlsysim.Systems.Clusters.Research_256

    # 2. Define the Solver
    solver = mlsysim.SustainabilityModel()

    # 3. Evaluate in Poland (Coal heavy grid)
    res_poland = solver.solve(
        fleet=fleet, 
        duration_days=30, 
        datacenter=mlsysim.Infra.Grids.Poland,
        mfu=0.45
    )
    
    # 4. Evaluate in Quebec (Hydroelectric heavy grid)
    res_quebec = solver.solve(
        fleet=fleet, 
        duration_days=30, 
        datacenter=mlsysim.Infra.Grids.Quebec,
        mfu=0.45
    )

    print(f"Total Energy Consumed: {res_poland.total_energy_kwh.to('MWh'):.1f}")
    print("-" * 40)
    print(f"Carbon Footprint (Poland): {res_poland.carbon_footprint_kg / 1000:,.1f} Tonnes")
    print(f"Carbon Footprint (Quebec): {res_quebec.carbon_footprint_kg / 1000:,.1f} Tonnes")
    print("-" * 40)
    
    ratio = res_poland.carbon_footprint_kg / res_quebec.carbon_footprint_kg
    print(f"The exact same training run produces {ratio:.1f}x more carbon in Poland than Quebec.")

if __name__ == "__main__":
    main()

# Expected output (mlsysim v0.1.1):
# Evaluating a 30-day training run on 256x H100 GPUs...
#
# Total Energy Consumed: 125.4 megawatt_hour
# ----------------------------------------
# Carbon Footprint (Poland): 102.8 Tonnes
# Carbon Footprint (Quebec): 1.7 Tonnes
# ----------------------------------------
# The exact same training run produces 61.1x more carbon in Poland than Quebec.
