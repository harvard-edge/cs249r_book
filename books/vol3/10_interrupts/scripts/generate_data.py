import os
import pandas as pd

def generate_compounding_error_data(output_path):
    """
    Generates data for the compounding error decay law: P_success = p^N.
    """
    print(f"Generating {output_path}...")
    horizons = list(range(1, 201))
    probabilities = [0.9, 0.95, 0.98, 0.99]
    
    data = {"Horizon_N": horizons}
    for p in probabilities:
        # Values are stored as percentages
        data[f"p_{p}"] = [ (p**n) * 100 for n in horizons ]
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

def generate_verification_tax_data(output_path):
    """
    Generates data for the Verification Tax Trade-off.
    
    Based on the notebook block "Where the verification tax curve bottoms out":
    Variables:
    N = 50 (Trajectory length)
    T_step = 1 s
    T_ver_heavy = 10 s (Full suite run)
    T_ver_fast = 0.05 s (Stage 2 static gate)
    p = 0.95 (Per-step semantic correctness)
    
    Formula:
    T(m) = (N / m) * ((m * T_step + T_ver) / (p^m))
    where m is the cadence K.
    """
    print(f"Generating {output_path}...")
    N = 50
    T_step = 1.0
    T_ver_heavy = 10.0
    T_ver_fast = 0.05
    p = 0.95
    
    cadences = list(range(1, 51))
    
    cost_heavy = []
    cost_fast = []
    
    for m in cadences:
        c_heavy = (N / m) * ((m * T_step + T_ver_heavy) / (p**m))
        c_fast = (N / m) * ((m * T_step + T_ver_fast) / (p**m))
        cost_heavy.append(c_heavy)
        cost_fast.append(c_fast)
        
    df = pd.DataFrame({
        "Cadence_K": cadences,
        "Cost_Heavy": cost_heavy,
        "Cost_Fast": cost_fast
    })
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    compounding_error_path = os.path.join(data_dir, "compounding_error.csv")
    verification_tax_path = os.path.join(data_dir, "verification_tax.csv")
    
    generate_compounding_error_data(compounding_error_path)
    generate_verification_tax_data(verification_tax_path)
    print("Data generation complete.")
