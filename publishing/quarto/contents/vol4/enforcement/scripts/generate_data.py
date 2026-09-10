import pandas as pd
import os

# Data based on industry milestones for fully autonomous driverless miles (e.g., Waymo and Cruise public reports)
# Provides proxy data for cumulative driverless miles across the industry over time.
data = [
    {"Date": "2021-01-01", "Miles_Millions": 0.1},
    {"Date": "2021-06-01", "Miles_Millions": 0.3},
    {"Date": "2022-01-01", "Miles_Millions": 0.5},
    {"Date": "2022-06-01", "Miles_Millions": 0.8},
    {"Date": "2023-01-01", "Miles_Millions": 1.0},
    {"Date": "2023-06-01", "Miles_Millions": 3.0},
    {"Date": "2023-12-01", "Miles_Millions": 7.0},
    {"Date": "2024-03-01", "Miles_Millions": 12.0},
    {"Date": "2024-06-01", "Miles_Millions": 22.0},
    {"Date": "2024-09-01", "Miles_Millions": 35.0},
    {"Date": "2024-12-01", "Miles_Millions": 50.0},
]

def main():
    df = pd.DataFrame(data)
    
    # Ensure the data directory exists
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Write to CSV
    output_path = os.path.join(output_dir, 'autonomous_miles.csv')
    df.to_csv(output_path, index=False)
    print(f"Successfully generated data at {output_path}")

if __name__ == '__main__':
    main()
