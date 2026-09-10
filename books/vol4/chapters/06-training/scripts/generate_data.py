import pandas as pd
import os

# Historical scaling of physics simulation throughput.
# These numbers represent approximate peak Steps Per Second (SPS) 
# achieved by various physics simulators in reinforcement learning contexts.
# Data sources are based on a synthesis of respective release notes, 
# whitepapers (e.g., Isaac Gym, Brax, MuJoCo MJX), and industry milestones.
DATA = [
    {
        "year": 2012,
        "simulator": "MuJoCo",
        "hardware": "CPU (1-core)",
        "architecture": "Sequential CPU",
        "sps": 2000,
        "source": "MuJoCo initial release (Todorov et al., 2012) - approximate single-thread SPS for complex robotic systems."
    },
    {
        "year": 2013,
        "simulator": "Bullet",
        "hardware": "CPU (1-core)",
        "architecture": "Sequential CPU",
        "sps": 1500,
        "source": "Bullet Physics 2.8x series - typical single-thread SPS for reinforcement learning environments."
    },
    {
        "year": 2017,
        "simulator": "MuJoCo (Ray)",
        "hardware": "CPU (32-core)",
        "architecture": "Distributed CPU",
        "sps": 50000,
        "source": "Ray architecture (Moritz et al., 2017) parallelizing MuJoCo instances across many CPU cores."
    },
    {
        "year": 2019,
        "simulator": "Isaac Gym Preview",
        "hardware": "NVIDIA Titan V",
        "architecture": "GPU Tensorized",
        "sps": 1000000,
        "source": "Isaac Gym Preview release (Makoviychuk et al., 2021) - first major tensorized GPU simulation."
    },
    {
        "year": 2021,
        "simulator": "Brax",
        "hardware": "Google TPUv3",
        "architecture": "TPU Tensorized",
        "sps": 30000000,
        "source": "Brax release (Freeman et al., 2021) - hardware-accelerated physics on JAX/TPU."
    },
    {
        "year": 2022,
        "simulator": "Isaac Gym Preview 4",
        "hardware": "NVIDIA A100",
        "architecture": "GPU Tensorized",
        "sps": 50000000,
        "source": "Isaac Gym optimizations on A100 architectures."
    },
    {
        "year": 2023,
        "simulator": "MuJoCo MJX",
        "hardware": "Google TPUv4/A100",
        "architecture": "TPU/GPU Tensorized",
        "sps": 100000000,
        "source": "MuJoCo MJX release by Google DeepMind - JAX-based port of MuJoCo."
    },
    {
        "year": 2024,
        "simulator": "Isaac Lab",
        "hardware": "NVIDIA H100",
        "architecture": "GPU Tensorized",
        "sps": 1000000000,
        "source": "Isaac Lab on Hopper architecture - peak throughput for highly parallelized multi-agent environments."
    }
]

def main():
    # Convert to DataFrame
    df = pd.DataFrame(DATA)
    
    # We only want the columns present in the original CSV
    columns_to_keep = ["year", "simulator", "hardware", "architecture", "sps"]
    df_csv = df[columns_to_keep]
    
    # Create the data directory if it doesn't exist
    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the output path
    output_path = os.path.join(output_dir, "simulation_throughput_scaling.csv")
    
    # Save the CSV without index
    df_csv.to_csv(output_path, index=False)
    print(f"Data successfully generated and saved to {output_path}")

if __name__ == "__main__":
    main()
