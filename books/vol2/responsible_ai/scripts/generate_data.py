import pandas as pd
import os

def main():
    """
    generate_data.py

    This script generates the proxy data for cumulative autonomous miles driven over time.
    The data is meant to illustrate the exponential growth in real-world deployment of autonomous 
    systems, emphasizing why safety must be treated as a fleet-level property.

    Data Source:
    The numbers are proxy data based on general industry milestones for autonomous vehicle testing 
    and deployment (e.g., early testing by companies like Waymo around 2012, scaling up 
    through the late 2010s, and significant driverless expansion in the 2020s). They are representative 
    estimates intended to show the scaling trend rather than precise records from a single company.
    """
    
    # Define the raw proxy data based on industry milestones
    data = [
        {"Year": 2012, "Cumulative_Miles_Millions": 0.3, "Notes": "Early testing"},
        {"Year": 2015, "Cumulative_Miles_Millions": 1.2, "Notes": ""},
        {"Year": 2016, "Cumulative_Miles_Millions": 2.0, "Notes": ""},
        {"Year": 2017, "Cumulative_Miles_Millions": 4.0, "Notes": ""},
        {"Year": 2018, "Cumulative_Miles_Millions": 10.0, "Notes": ""},
        {"Year": 2020, "Cumulative_Miles_Millions": 20.0, "Notes": ""},
        {"Year": 2021, "Cumulative_Miles_Millions": 25.0, "Notes": ""},
        {"Year": 2022, "Cumulative_Miles_Millions": 32.0, "Notes": ""},
        {"Year": 2023, "Cumulative_Miles_Millions": 45.0, "Notes": ""},
        {"Year": 2024, "Cumulative_Miles_Millions": 80.0, "Notes": "Driverless expansion"},
    ]

    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Determine output path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "data")
    output_file = os.path.join(output_dir, "autonomous_miles_scaling.csv")
    
    # Ensure data directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the dataframe to CSV without index
    df.to_csv(output_file, index=False)
    print(f"Successfully generated data at: {output_file}")

if __name__ == "__main__":
    main()
