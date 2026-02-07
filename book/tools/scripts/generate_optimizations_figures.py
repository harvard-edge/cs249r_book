
import sys
import os
import matplotlib.pyplot as plt

# Setup path to import from quarto/physx
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'quarto'))

from physx import viz
from physx import ch_optimizations

def generate_assets():
    output_dir = os.path.join(project_root, "quarto/assets/images/generated/optimizations")
    os.makedirs(output_dir, exist_ok=True)
    
    viz.set_book_style()
    
    print("Generating AlexNet Filters...")
    fig, ax = plt.subplots(figsize=(8, 8))
    ch_optimizations.plot_alexnet_filters(ax)
    plt.savefig(os.path.join(output_dir, "alexnet_filters.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Generating Sparsity Heatmap (Mock Data)...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ch_optimizations.plot_sparsity_heatmap(ax)
    plt.savefig(os.path.join(output_dir, "sparsity_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"All assets generated in {output_dir}")

if __name__ == "__main__":
    generate_assets()
