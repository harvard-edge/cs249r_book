import sys
import os
import matplotlib.pyplot as plt

# Add calc directory to path
sys.path.append(os.path.abspath('book/quarto/calc'))
import viz

# Output directory
OUTPUT_DIR = "book/quarto/assets/preview_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
viz.set_book_style()

print("Generating ML Lifecycle...")
viz.plot_ml_lifecycle()
plt.savefig(f"{OUTPUT_DIR}/ml_lifecycle.png")
plt.close('all')

print("Generating Distributed Training...")
viz.plot_distributed_training()
plt.savefig(f"{OUTPUT_DIR}/distributed_training.png")
plt.close('all')

print(f"Plots saved to {OUTPUT_DIR}")
