import pandas as pd
import os

# Data generation logic based on the Sovereign Law of Agency:
# Unverified trajectory success decays exponentially with horizon length H.
# P(tau) = (1 - epsilon * (1 - v))^H
# where:
# - H is the Horizon (number of steps)
# - epsilon = (1 - p) is the step error rate (p is per-step accuracy)
# - v is the verification precision

def calculate_probability(p, v, horizon_max=100):
    epsilon = 1.0 - p
    step_success = 1.0 - (epsilon * (1.0 - v))
    return [step_success ** h for h in range(1, horizon_max + 1)]

horizons = list(range(1, 101))

data = {
    "Horizon": horizons,
    "Unverified_95": calculate_probability(p=0.95, v=0.0),
    "Unverified_98": calculate_probability(p=0.98, v=0.0),
    "Unverified_99": calculate_probability(p=0.99, v=0.0),
    "Verified_98_v70": calculate_probability(p=0.98, v=0.7),
    "Verified_98_v90": calculate_probability(p=0.98, v=0.9)
}

df = pd.DataFrame(data)

# Save the DataFrame to CSV
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "trajectory_decay.csv")

df.to_csv(csv_path, index=False)
print(f"Data successfully generated and written to {csv_path}")
