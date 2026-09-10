import pandas as pd
import os

def generate_bandwidth_trends():
    """
    Generate bandwidth_trends.csv which tracks the historical bandwidth of 
    intra-node and inter-node interconnect technologies.
    """
    
    # Data sourced from standard interconnect specifications and NVIDIA datasheets.
    # Intra-node bandwidth represents aggregate bidirectional bandwidth per GPU.
    # Inter-node bandwidth represents per-port unidirectional bandwidth in GB/s.
    
    data = [
        # Intra-node (PCIe / NVLink)
        # PCIe 3.0 x16 provides ~15.75 GB/s, typically rounded to 16 GB/s
        {"Year": 2012, "Technology": "PCIe 3.0", "Type": "Intra-node", "Bandwidth_GBps": 16},
        # NVLink 1.0 (Pascal P100): 4 links * 40 GB/s = 160 GB/s bidirectional
        {"Year": 2016, "Technology": "NVLink 1.0", "Type": "Intra-node", "Bandwidth_GBps": 160},
        # NVLink 2.0 (Volta V100): 6 links * 50 GB/s = 300 GB/s bidirectional
        {"Year": 2017, "Technology": "NVLink 2.0", "Type": "Intra-node", "Bandwidth_GBps": 300},
        # NVLink 3.0 (Ampere A100): 12 links * 50 GB/s = 600 GB/s bidirectional
        {"Year": 2020, "Technology": "NVLink 3.0", "Type": "Intra-node", "Bandwidth_GBps": 600},
        # NVLink 4.0 (Hopper H100): 18 links * 50 GB/s = 900 GB/s bidirectional
        {"Year": 2022, "Technology": "NVLink 4.0", "Type": "Intra-node", "Bandwidth_GBps": 900},
        # NVLink 5.0 (Blackwell B100/B200): 18 links * 100 GB/s = 1800 GB/s bidirectional
        {"Year": 2024, "Technology": "NVLink 5.0", "Type": "Intra-node", "Bandwidth_GBps": 1800},
        
        # Inter-node (InfiniBand)
        # FDR: 56 Gbps -> 7 GB/s
        {"Year": 2012, "Technology": "InfiniBand FDR", "Type": "Inter-node", "Bandwidth_GBps": 7},
        # EDR: 100 Gbps -> 12.5 GB/s
        {"Year": 2014, "Technology": "InfiniBand EDR", "Type": "Inter-node", "Bandwidth_GBps": 12.5},
        # HDR: 200 Gbps -> 25 GB/s
        {"Year": 2017, "Technology": "InfiniBand HDR", "Type": "Inter-node", "Bandwidth_GBps": 25},
        # NDR: 400 Gbps -> 50 GB/s
        {"Year": 2021, "Technology": "InfiniBand NDR", "Type": "Inter-node", "Bandwidth_GBps": 50},
        # XDR: 800 Gbps -> 100 GB/s
        {"Year": 2024, "Technology": "InfiniBand XDR", "Type": "Inter-node", "Bandwidth_GBps": 100},
    ]

    df = pd.DataFrame(data)
    
    # Define output path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "../data/bandwidth_trends.csv")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export to CSV without index
    # Note on types: Bandwidth_GBps becomes float because of 12.5. We will format it appropriately if needed, 
    # but pandas to_csv naturally handles it and writes 16 as 16, 12.5 as 12.5 etc.
    df['Bandwidth_GBps'] = df['Bandwidth_GBps'].apply(lambda x: str(int(x)) if x == int(x) else str(x))
    df.to_csv(output_path, index=False)
    print(f"Data successfully generated and saved to {output_path}")

if __name__ == "__main__":
    generate_bandwidth_trends()
