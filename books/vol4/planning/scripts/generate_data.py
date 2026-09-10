import pandas as pd
import os

def generate_autonomous_miles_data():
    """
    Generates a dataset representing the cumulative autonomous miles driven in the wild.
    
    Data Source / Logic:
    This dataset contains proxy data based on industry milestones and public reporting 
    by major autonomous vehicle companies (such as Waymo and Cruise). The values 
    approximate the exponential growth trend in real-world autonomous testing and 
    deployment from 2015 to 2024, representing millions of cumulative miles.
    
    The resulting CSV is used in the 'Autonomous Miles Scaling' figure to illustrate 
    the rapid scale-up of autonomous trajectory planning requirements.
    """
    
    # Define the data based on historical industry milestone trends
    data = {
        "Year": [
            2015,
            2016,
            2017,
            2018,
            2019,
            2020,
            2021,
            2022,
            2023,
            2024
        ],
        "Cumulative_Miles_Millions": [
            1.2,
            2.5,
            4.0,
            10.0,
            20.0,
            25.0,
            30.0,
            40.0,
            55.0,
            80.0
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Define the output path relative to the script location
    # The script is in `scripts/`, so we go up one directory to reach `data/`
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "data")
    output_path = os.path.join(output_dir, "autonomous_miles.csv")
    
    # Ensure the data directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the dataframe to CSV, keeping the exact format
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_autonomous_miles_data()
