import pandas as pd

def generate_reliability_data():
    """
    Generates data illustrating the 'Reliability Wall' in agentic systems.
    
    The data models the trajectory success probability of an agentic system over multiple steps.
    According to the compounding reliability equation: P_success = p^N
    where:
    - N is the Trajectory Length (number of sequential steps)
    - p is the Step Accuracy (probability of success for a single step)
    
    We calculate the success probability for different step accuracies (0.8, 0.9, 0.95, 0.99)
    across trajectory lengths from 1 to 50 steps.
    
    The values represent theoretical upper bounds assuming independent failure probabilities
    at each step.
    """
    
    data = []
    
    # Range of trajectory lengths from 1 to 50 steps
    trajectory_lengths = range(1, 51)
    
    # Various single-step accuracies (e.g., 80%, 90%, 95%, 99%)
    step_accuracies = [0.8, 0.9, 0.95, 0.99]
    
    for N in trajectory_lengths:
        for p in step_accuracies:
            # Theoretical probability of success for N steps with accuracy p
            success_prob = p ** N
            
            data.append({
                'Trajectory Length (N)': N,
                'Step Accuracy (p)': p,
                'Success Probability': success_prob
            })
            
    df = pd.DataFrame(data)
    
    # The output is saved to the data/ directory for visualization
    output_path = '../data/reliability_wall.csv'
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {output_path}")

if __name__ == '__main__':
    generate_reliability_data()
