import numpy as np
import pandas as pd
import os

# Data sourced from Section 17.2: What the Method Cannot Establish
# The formula calculates the required number of independent, failure-free operational hours (n)
# to assert with a given confidence level (1 - alpha) that the true failure rate does not exceed p.
# Formula: n >= ln(alpha) / ln(1 - p)
# We use alpha = 0.05 for a 95% confidence level.

def generate_exposure_scaling_data():
    alpha = 0.05  # 95% confidence level
    # Target failure probability per hour (p), ranging from 1e-2 to 1e-10
    p_values = np.logspace(-2, -10, 100)
    
    # Calculate required exposure hours
    n_values = np.log(alpha) / np.log(1 - p_values)
    
    # Calculate machine years (assuming 24 hours/day, 365.25 days/year)
    machine_years = n_values / (24 * 365.25)
    
    df = pd.DataFrame({
        'target_failure_rate': p_values,
        'required_exposure_hours': n_values,
        'machine_years': machine_years
    })
    
    # Save to data/exposure_scaling.csv
    # Script is in scripts/ directory, so we go up one level
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', 'data', 'exposure_scaling.csv')
    
    df.to_csv(output_path, index=False)
    print(f"Data successfully generated and saved to {output_path}")

if __name__ == "__main__":
    generate_exposure_scaling_data()
