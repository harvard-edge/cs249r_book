import pandas as pd
import os

# Data sourced from California DMV Autonomous Vehicle Disengagement Reports
# Representative trend data for Miles per Disengagement for leading AV companies (e.g., Waymo)
# from 2015 to 2021.
data = {
    "Year": [2015, 2016, 2017, 2018, 2019, 2020, 2021],
    "Miles_per_Disengagement": [1224, 5128, 5596, 11017, 13219, 29945, 59991]
}

def main():
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Output file path relative to the script directory
    output_dir = os.path.join(script_dir, "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "av_disengagement_trends.csv")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Successfully generated {output_file}")

if __name__ == "__main__":
    main()
