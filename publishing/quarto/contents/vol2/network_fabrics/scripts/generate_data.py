import pandas as pd
import os

def generate_bandwidth_trends():
    """
    Generate the bandwidth_trends.csv data.
    
    Data sources and methodology:
    - Ethernet: Proxy dates for 40G, 100G, 400G, 800G standard releases/adoption in data centers.
    - InfiniBand: Trade association specs release dates: FDR (56G, 2011), EDR (100G, 2014), HDR (200G, 2017), NDR (400G, 2021), XDR (800G, 2024).
    - NVLink: Bidirectional total bandwidth converted to Gbps (1 GB/s = 8 Gbps) based on NVIDIA specs.
        - v1 (Pascal, 2016): 160 GB/s = 1280 Gbps
        - v2 (Volta, 2017): 300 GB/s = 2400 Gbps
        - v3 (Ampere, 2020): 600 GB/s = 4800 Gbps
        - v4 (Hopper, 2022): 900 GB/s = 7200 Gbps
        - v5 (Blackwell, 2024): 1800 GB/s = 14400 Gbps
    - PCIe (x16): Approximate raw bandwidth based on PCI-SIG release dates.
        - Gen 3 (2010): ~16 GB/s = 128 Gbps
        - Gen 4 (2017): ~32 GB/s = 256 Gbps
        - Gen 5 (2019): ~64 GB/s = 512 Gbps
        - Gen 6 (2022): ~128 GB/s = 1024 Gbps
    """
    
    data = [
        # Ethernet
        {'Year': 2010, 'Technology': 'Ethernet', 'Bandwidth_Gbps': 40},
        {'Year': 2015, 'Technology': 'Ethernet', 'Bandwidth_Gbps': 100},
        {'Year': 2019, 'Technology': 'Ethernet', 'Bandwidth_Gbps': 400},
        {'Year': 2023, 'Technology': 'Ethernet', 'Bandwidth_Gbps': 800},
        
        # InfiniBand
        {'Year': 2011, 'Technology': 'InfiniBand', 'Bandwidth_Gbps': 56},
        {'Year': 2014, 'Technology': 'InfiniBand', 'Bandwidth_Gbps': 100},
        {'Year': 2017, 'Technology': 'InfiniBand', 'Bandwidth_Gbps': 200},
        {'Year': 2021, 'Technology': 'InfiniBand', 'Bandwidth_Gbps': 400},
        {'Year': 2024, 'Technology': 'InfiniBand', 'Bandwidth_Gbps': 800},
        
        # NVLink
        {'Year': 2016, 'Technology': 'NVLink', 'Bandwidth_Gbps': 1280},
        {'Year': 2017, 'Technology': 'NVLink', 'Bandwidth_Gbps': 2400},
        {'Year': 2020, 'Technology': 'NVLink', 'Bandwidth_Gbps': 4800},
        {'Year': 2022, 'Technology': 'NVLink', 'Bandwidth_Gbps': 7200},
        {'Year': 2024, 'Technology': 'NVLink', 'Bandwidth_Gbps': 14400},
        
        # PCIe (x16)
        {'Year': 2010, 'Technology': 'PCIe (x16)', 'Bandwidth_Gbps': 128},
        {'Year': 2017, 'Technology': 'PCIe (x16)', 'Bandwidth_Gbps': 256},
        {'Year': 2019, 'Technology': 'PCIe (x16)', 'Bandwidth_Gbps': 512},
        {'Year': 2022, 'Technology': 'PCIe (x16)', 'Bandwidth_Gbps': 1024},
    ]
    
    df = pd.DataFrame(data)
    
    # Save to data directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.abspath(os.path.join(output_dir, 'bandwidth_trends.csv'))
    
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {output_path}")

if __name__ == '__main__':
    generate_bandwidth_trends()
