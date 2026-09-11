import pandas as pd
import os

# Data sourced as an industry proxy for cumulative autonomous miles driven over time.
# Represents milestones from companies developing autonomous vehicle fleets (e.g., Waymo, Tesla).
data = {
    "Year": [
        2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 
        2019, 2020, 2021, 2022, 2023, 2024
    ],
    "Cumulative_Miles_Millions": [
        0.01, 0.14, 0.20, 0.30, 0.50, 0.70, 1.2, 2.0, 4.0, 10.0, 
        20.0, 30.0, 45.0, 65.0, 80.0, 120.0
    ]
}

def main():
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Resolve the path to the data directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, "autonomous_miles_scaling.csv")
    df.to_csv(output_path, index=False)
    print(f"Data successfully generated and saved to {output_path}")

if __name__ == "__main__":
    main()
