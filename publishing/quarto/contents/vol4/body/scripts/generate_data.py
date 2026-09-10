import pandas as pd
import os

"""
Generates the autonomous_miles_scaling.csv dataset for the scaling of physical AI plot.
Data represents approximate cumulative autonomous miles driven over time (in millions),
primarily tracking major milestones from industry leaders like Waymo.

Sources/Milestones:
- 2009: Google self-driving car project begins (0 miles)
- 2012: 300,000 miles announced (0.3M)
- 2014: 700,000 miles (0.7M)
- 2015: 1.2 million miles
- 2016: 2 million miles
- 2017: 4 million miles
- 2018: 10 million miles
- 2019: 20 million miles (Waymo milestone)
- 2020: 25 million miles 
- 2021: 30 million miles 
- 2023: 50 million miles 
- 2024: 71 million miles 
"""

def generate_data():
    data = {
        "Year": [2009, 2012, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2023, 2024],
        "Cumulative_Miles_Millions": [0.0, 0.3, 0.7, 1.2, 2.0, 4.0, 10.0, 20.0, 25.0, 30.0, 50.0, 71.0]
    }
    
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "../data")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "autonomous_miles_scaling.csv")
    df.to_csv(output_path, index=False)
    print(f"Data successfully generated and saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_data()
