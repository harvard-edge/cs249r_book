import pandas as pd
import os

def generate_robot_dataset_scaling():
    """
    Generates the robot_dataset_scaling.csv file which tracks the historical scaling
    of physical robot learning datasets over time.
    
    Data Sources:
    - 2016: Levine et al. (Arm Farm) - "Learning Hand-Eye Coordination for Robotic Grasping..." 
      (~800,000 grasp attempts across 14 manipulators)
    - 2018: QT-Opt - "Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation" 
      (~580,000 offline grasp attempts)
    - 2019: RoboNet - "RoboNet: Large-Scale Multi-Robot Learning"
      (162,000 trajectories across 7 robot platforms)
    - 2021: Bridge Data - "Bridge Data: Boosting Generalization of Robotic Skills..."
      (7,200 demonstrations)
    - 2022: RT-1 - "RT-1: Robotics Transformer for Real-World Control at Scale"
      (~130,000 episodes on 13 Everyday Robots)
    - 2023: Open X-Embodiment - "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"
      (~2,000,000+ trajectories across 22 robot embodiments)
    - 2024: DROID - "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset"
      (76,000 trajectories for Franka)
    - 2024: RH20T - "RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills"
      (~110,000 robot trajectories)
    """

    data = [
        {
            "Year": 2016,
            "Dataset": "Levine et al. (Arm Farm)",
            "Episodes": 800000,
            "Embodiments": 1,
            "Modality": "Vision-Action",
            "Notes": "14 robots, 2 months, single task (grasping)"
        },
        {
            "Year": 2018,
            "Dataset": "QT-Opt",
            "Episodes": 580000,
            "Embodiments": 1,
            "Modality": "Vision-Action",
            "Notes": "Continuous RL, single task (grasping)"
        },
        {
            "Year": 2019,
            "Dataset": "RoboNet",
            "Episodes": 162000,
            "Embodiments": 7,
            "Modality": "Vision-Action",
            "Notes": "First large-scale cross-embodiment video dataset"
        },
        {
            "Year": 2021,
            "Dataset": "Bridge Data",
            "Episodes": 7200,
            "Embodiments": 1,
            "Modality": "Vision-Action",
            "Notes": "Multi-task, single robot, simulated kitchen"
        },
        {
            "Year": 2022,
            "Dataset": "RT-1",
            "Episodes": 130000,
            "Embodiments": 1,
            "Modality": "Vision-Action",
            "Notes": "High task diversity, single robot (Everyday Robots)"
        },
        {
            "Year": 2023,
            "Dataset": "Open X-Embodiment",
            "Episodes": 2000000,
            "Embodiments": 22,
            "Modality": "Vision-Action",
            "Notes": "Massive multi-institutional cross-embodiment (RT-X)"
        },
        {
            "Year": 2024,
            "Dataset": "DROID",
            "Episodes": 76000,
            "Embodiments": 1,
            "Modality": "Vision-Action",
            "Notes": "Distributed robot manipulation dataset (Franka)"
        },
        {
            "Year": 2024,
            "Dataset": "RH20T",
            "Episodes": 110000,
            "Embodiments": 1,
            "Modality": "Vision-Action",
            "Notes": "Robotic manipulation dataset with complex skills"
        }
    ]

    df = pd.DataFrame(data)
    
    # Ensure the data directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "robot_dataset_scaling.csv")
    df.to_csv(output_file, index=False)
    print(f"Successfully generated {output_file}")

if __name__ == "__main__":
    generate_robot_dataset_scaling()
