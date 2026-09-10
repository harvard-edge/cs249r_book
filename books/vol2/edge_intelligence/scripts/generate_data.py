import pandas as pd
import os

def generate_ane_trend():
    """
    Generate the Apple Neural Engine (ANE) trend data (TOPS) over the years.
    Data is sourced from Apple's official specifications and keynotes
    for their A-series Bionic and Pro chips.
    """
    data = [
        {"Year": 2017, "Chip": "A11 Bionic", "TOPS": 0.6},
        {"Year": 2018, "Chip": "A12 Bionic", "TOPS": 5.0},
        {"Year": 2019, "Chip": "A13 Bionic", "TOPS": 6.0},
        {"Year": 2020, "Chip": "A14 Bionic", "TOPS": 11.0},
        {"Year": 2021, "Chip": "A15 Bionic", "TOPS": 15.8},
        {"Year": 2022, "Chip": "A16 Bionic", "TOPS": 17.0},
        {"Year": 2023, "Chip": "A17 Pro", "TOPS": 35.0},
        {"Year": 2024, "Chip": "A18 Pro", "TOPS": 35.0},
    ]

    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs('../data', exist_ok=True)
    
    # Save to CSV
    output_path = '../data/ane_trend.csv'
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    # Get absolute path of this script's directory to resolve paths reliably
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    generate_ane_trend()
