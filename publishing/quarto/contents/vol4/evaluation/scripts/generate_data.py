import pandas as pd
import os

def generate_autonomous_miles_scaling_data():
    """
    Generates the autonomous_miles_scaling.csv dataset.
    
    Data Source: Proxy data based on industry milestones for leading autonomous 
    vehicle programs (e.g., Waymo, Cruise) as they scaled operations.
    The values represent cumulative miles (in millions).
    Early years do not have reported simulated miles at scale.
    """
    data = {
        "Year": [2010, 2012, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        "Physical_Miles": [0.01, 0.3, 0.7, 1.0, 2.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 71.0, 100.0],
        "Simulated_Miles": [None, None, None, 1000.0, 2500.0, 5000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0, 40000.0, 50000.0]
    }

    df = pd.DataFrame(data)
    
    # Define output path relative to this script
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "autonomous_miles_scaling.csv")
    df.to_csv(output_file, index=False)
    print(f"Data successfully generated and saved to {output_file}")

if __name__ == "__main__":
    generate_autonomous_miles_scaling_data()
