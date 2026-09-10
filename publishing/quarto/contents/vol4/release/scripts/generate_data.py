import pandas as pd
import numpy as np
import os

def generate_testing_gap_data():
    """
    Generates data for the 'Autonomy Testing Gap' figure.
    This demonstrates the time required (in years) of continuous failure-free
    testing needed to prove a target hazard rate at a 95% confidence level.
    """
    # 50 points from 10^-3 to 10^-8, typical hazard rate targets for 
    # safety-critical systems (e.g., aerospace, autonomous vehicles)
    target_hazard_rates = np.logspace(-3, -8, 50)
    
    # Calculate required continuous testing years with 0 failures to reach 95% confidence.
    # From the Poisson/Exponential distribution reliability formula: N = -ln(1 - confidence) / lambda
    # Here, confidence = 0.95 -> -ln(0.05) ~= 2.9957
    # Divide by 8760 (hours in a year) to get the time in years.
    hours_in_year = 8760
    confidence_level = 0.95
    n_failures_multiplier = -np.log(1 - confidence_level)
    
    black_box_years = n_failures_multiplier / (target_hazard_rates * hours_in_year)
    
    # Architectural Enforcer Multiplier
    # When using a hardware-isolated safety enforcer, the overall system hazard rate is:
    # lambda_system = lambda_policy * (1 - c) + lambda_enforcer
    # We solve for lambda_policy:
    # lambda_policy = (lambda_system - lambda_enforcer) / (1 - c)
    #
    # Note: The .qmd chapter text conceptually states an enforcer failure rate of 10^-6,
    # but the actual chart data uses an enforcer failure rate of 10^-9. We use 10^-9 
    # here to exactly reproduce the historical data.
    enforcer_coverage = 0.9999
    enforcer_failure_rate = 1e-9
    
    policy_hazard_rates = (target_hazard_rates - enforcer_failure_rate) / (1 - enforcer_coverage)
    
    architectural_years = n_failures_multiplier / (policy_hazard_rates * hours_in_year)
    
    df = pd.DataFrame({
        "Target_Hazard_Rate": target_hazard_rates,
        "Black_Box_Years": black_box_years,
        "Architectural_Years": architectural_years
    })
    
    # Save the dataframe to the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "testing_gap.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data successfully generated and saved to {csv_path}")

if __name__ == "__main__":
    generate_testing_gap_data()
