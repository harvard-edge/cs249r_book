import pandas as pd
import os

# Data representing the number of Adversarial ML papers published on arXiv per year.
# These numbers are proxy data illustrating the exponential growth and the "Adversarial Arms Race",
# reflecting the intense cycle of proposing new defenses and subsequent adversarial attacks breaking them.
# The data is based on representative trends in the ML security literature between 2014 and 2023.
raw_data = {
    "Year": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "Papers_Published": [5, 20, 80, 250, 650, 1200, 1800, 2400, 2900, 3500]
}

def generate_csv():
    # Create DataFrame
    df = pd.DataFrame(raw_data)
    
    # Calculate cumulative papers
    df["Cumulative_Papers"] = df["Papers_Published"].cumsum()
    
    # Ensure the target directory exists
    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, "adv_ml_papers_arxiv.csv")
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_csv()
