import pandas as pd
import numpy as np
import os

def generate_volumetric_memory_scaling():
    """
    Generates the spatial memory scaling dataset (volumetric_memory_scaling.csv).
    
    Data rationale:
    - Resolutions (N) range from 128 to 8192 voxels per dimension.
    - Dense Voxel Grid: O(N^3) scaling. Assuming 4 bytes per voxel (e.g., float32 TSDF or int32 occupancy).
      Footprint = (N^3) * 4 bytes / 10^9 (for GB).
    - Hierarchical Octree: O(N^2) scaling for mostly empty space (surface-level occupancy). 
      Assuming 32 bytes per surface node (child pointers + data).
      Footprint = (N^2) * 32 bytes / 10^9 (for GB).
    - Continuous 3DGS (3D Gaussian Splatting): O(K) scaling. Independent of strict geometric grid quantization.
      Remains constant (represented as 1.0 GB) across grid resolutions since footprint is dictated by 
      scene complexity (number of Gaussians), not grid resolution.
    """
    resolutions = [128, 256, 512, 1024, 2048, 4096, 8192]
    
    # Calculate footprints in GB (using 1 GB = 10^9 bytes as per original data)
    dense_grid_gb = [(n**3) * 4 / 1e9 for n in resolutions]
    octree_gb = [(n**2) * 32 / 1e9 for n in resolutions]
    continuous_3dgs_gb = [1.0 for _ in resolutions]
    
    df = pd.DataFrame({
        'Resolution': resolutions,
        'Dense_Grid_GB': dense_grid_gb,
        'Octree_GB': octree_gb,
        'Continuous_3DGS_GB': continuous_3dgs_gb
    })
    
    # Format to preserve the exact same formatting
    # Actually, pandas to_csv will use sufficient precision, but let's ensure no rounding mismatch.
    # The original data has up to 9 decimal places.
    
    # Determine absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, 'volumetric_memory_scaling.csv')
    df.to_csv(csv_path, index=False)
    print(f"Successfully generated {csv_path}")

if __name__ == '__main__':
    generate_volumetric_memory_scaling()
