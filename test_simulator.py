# test_simulator.py
import sys
import os

# No more sys.path hacks needed if we pip install -e . or run from root
try:
    import mlsysim
except ImportError:
    # Fallback for local run without install
    sys.path.insert(0, ".")
    import mlsysim

from mlsysim.sim import ResourceSimulation
from mlsysim.book import Applications

def run_poc():
    print("--- 🚀 MLSysim: Root Package Verification ---")
    
    # 1. Setup
    scenario = Applications.Doorbell
    persona = mlsysim.sim.Personas.TinyPioneer
    
    print(f"Branding:  mlsysim (Root Namespace)")
    print(f"Persona:   {persona.role}")
    print("-" * 40)

    # 2. Run
    sim_engine = ResourceSimulation(scenario, persona)
    ledger = sim_engine.evaluate({"region": "Quebec"})

    print(f"Annual Carbon: {ledger.sustainability.carbon_kg:,.0f} kg CO2e")
    
    if ledger.sustainability.carbon_kg > 0:
        print("-" * 40)
        print(f"✅ Root Package Structure working perfectly!")
        print("-" * 40)

if __name__ == "__main__":
    run_poc()
