import sys
import os
import matplotlib.pyplot as plt

# Add physx directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'quarto'))
from physx import viz

# Output directory
OUTPUT_DIR = "book/quarto/assets/preview_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
viz.set_book_style()

print("Generating Systems Gap...")
viz.plot_systems_gap()
plt.savefig(f"{OUTPUT_DIR}/systems_gap.png")
plt.close('all')

print(f"Plot saved to {OUTPUT_DIR}/systems_gap.png")
