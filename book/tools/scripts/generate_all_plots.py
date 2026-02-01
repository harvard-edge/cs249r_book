
import sys
import os
import matplotlib.pyplot as plt

# Setup path to import from quarto/calc
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'quarto'))

from calc import viz
from calc import ch_hw_acceleration
from calc import ch_appendix_algorithm

def generate_preview_assets():
    output_dir = os.path.join(project_root, "quarto/assets/images/generated")
    os.makedirs(output_dir, exist_ok=True)
    
    viz.set_book_style()
    
    print("Generating Technology S-Curve...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ch_hw_acceleration.plot_technology_s_curve(ax)
    plt.savefig(os.path.join(output_dir, "technology_s_curve.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Generating Broadcasting Visual...")
    fig, ax = plt.subplots(figsize=(12, 5))
    ch_appendix_algorithm.plot_broadcasting_visual(ax)
    plt.savefig(os.path.join(output_dir, "broadcasting_python.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"All assets generated in {output_dir}")

if __name__ == "__main__":
    generate_preview_assets()
