"""
Sustainability Lab: Carbon-Aware Fleet Design
=============================================
This lab teaches students how to model the 'Hierarchy of Environment'
by comparing the same GPU fleet across different regional grids.
"""

import mlsysim
from mlsysim.infra.types import Datacenter

def main():
    print("Scenario: Training Llama-3-70B on 512 H100 GPUs for 30 days\n")
    
    # 1. Setup the Fleet
    node = mlsysim.Systems.Nodes.DGX_H100
    fleet = mlsysim.Fleet(
        name="Frontier Training Cluster",
        node=node,
        count=64, # 64 nodes * 8 GPUs = 512 GPUs
        fabric=mlsysim.Systems.Fabrics.InfiniBand_NDR
    )
    
    # 2. Define our Experimental Regions
    experiments = [
        {"name": "Poland (Coal-Heavy)", "grid": mlsysim.Infra.Grids.Poland},
        {"name": "Quebec (Hydro-Clean)", "grid": mlsysim.Infra.Grids.Quebec}
    ]
    
    print(f"{'Region':<25} | {'PUE':<6} | {'Energy (MWh)':<12} | {'Carbon (Tonnes)':<12}")
    print("-" * 65)
    
    solver = mlsysim.SustainabilityModel()
    
    for exp in experiments:
        # We'll assume a liquid-cooled profile override
        dc = Datacenter(name="Custom DC", grid=exp['grid'], pue_override=1.06)
        
        impact = solver.solve(fleet, duration_days=30, datacenter=dc)
        
        energy_mwh = impact.total_energy_kwh.m_as('megawatt_hour')
        carbon_tonnes = impact.carbon_footprint_kg / 1000.0
        
        print(f"{exp['name']:<25} | {dc.pue:<6.2f} | {energy_mwh:<12.1f} | {carbon_tonnes:<12.1f}")

    print("\nConclusion: Moving the same hardware to a cleaner grid reduces carbon by >90%.")

if __name__ == "__main__":
    main()

# Expected output (mlsysim v0.1.1):
# Scenario: Training Llama-3-70B on 512 H100 GPUs for 30 days
#
# Region                    | PUE    | Energy (MWh) | Carbon (Tonnes)
# -----------------------------------------------------------------
# Poland (Coal-Heavy)       | 1.06   | 273.5        | 224.3
# Quebec (Hydro-Clean)      | 1.06   | 273.5        | 5.5
#
# Conclusion: Moving the same hardware to a cleaner grid reduces carbon by >90%.
