import pandas as pd
import os

def generate_rlvr_data():
    """
    Generates proxy data for the 'Emergent Reasoning under Pure RLVR' figure.
    This data illustrates the 'aha moment' and test-time compute scaling characteristic 
    of pure RL training on verifiable rewards (e.g., DeepSeek-R1-Zero).
    
    The data represents an S-curve where pass rates and average reasoning tokens 
    initially stay low, hit an inflection point where the model discovers test-time 
    compute directly increases success rate, and then asymptotes.
    """
    
    # Stylized proxy data based on emergent reasoning patterns in RLVR
    data = {
        'training_step': [
            0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 
            5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000
        ],
        'pass_rate': [
            3.2, 4.1, 5.3, 6.8, 12.4, 22.1, 38.5, 56.2, 71.4, 81.3, 
            86.5, 88.9, 90.2, 90.8, 91.3, 91.6, 91.8, 92.0, 92.1, 92.2, 92.3
        ],
        'avg_reasoning_tokens': [
            110, 118, 132, 175, 380, 840, 1520, 2280, 2890, 3240, 
            3410, 3500, 3560, 3600, 3630, 3650, 3665, 3675, 3685, 3695, 3700
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Ensure the output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'rlvr_emergent_reasoning.csv')
    df.to_csv(output_path, index=False)
    print(f"Successfully generated data at {output_path}")

if __name__ == '__main__':
    generate_rlvr_data()
