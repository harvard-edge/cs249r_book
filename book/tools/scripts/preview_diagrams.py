import sys
import os
import matplotlib.pyplot as plt

# Add mlsys directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'quarto'))
from mlsys import viz

# Output directory
OUTPUT_DIR = "book/quarto/assets/preview_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_ml_lifecycle(ax=None):
    """Visualizes 'ML System Lifecycle' (Circular Flow)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    from matplotlib.patches import FancyBboxPatch, ConnectionPatch

    # Define Nodes (Circular layout roughly)
    nodes = {
        'Collection': {'x': 2, 'y': 6, 'label': 'Data\nCollection', 'color': viz.COLORS['BlueL'], 'edge': viz.COLORS['BlueLine']},
        'Prep':       {'x': 6, 'y': 6, 'label': 'Data\nPreparation', 'color': viz.COLORS['GreenL'], 'edge': viz.COLORS['GreenLine']},
        'Train':      {'x': 10, 'y': 6, 'label': 'Model\nTraining', 'color': viz.COLORS['OrangeL'], 'edge': viz.COLORS['OrangeLine']},
        'Eval':       {'x': 10, 'y': 2, 'label': 'Model\nEvaluation', 'color': viz.COLORS['RedL'], 'edge': viz.COLORS['RedLine']},
        'Deploy':     {'x': 6, 'y': 2, 'label': 'Model\nDeployment', 'color': viz.COLORS['VioletL'], 'edge': viz.COLORS['VioletLine']},
        'Monitor':    {'x': 2, 'y': 2, 'label': 'Model\nMonitoring', 'color': viz.COLORS['OrangeL'], 'edge': viz.COLORS['OrangeLine']},
    }

    # Draw Nodes
    for _, node in nodes.items():
        p = FancyBboxPatch((node['x']-0.9, node['y']-0.6), 1.8, 1.2, boxstyle="round,pad=0.1",
                           fc=node['color'], ec=node['edge'], linewidth=2)
        ax.add_patch(p)
        ax.text(node['x'], node['y'], node['label'], ha='center', va='center', fontsize=9, fontweight='bold')

    # Draw Arrows (Main Cycle)
    arrows = [
        ('Collection', 'Prep'), ('Prep', 'Train'), ('Train', 'Eval'),
        ('Eval', 'Deploy'), ('Deploy', 'Monitor'), ('Monitor', 'Collection')
    ]

    for start, end in arrows:
        con = ConnectionPatch(xyA=(nodes[start]['x'], nodes[start]['y']), xyB=(nodes[end]['x'], nodes[end]['y']),
                              coordsA="data", coordsB="data",
                              axesA=ax, axesB=ax,
                              arrowstyle="-|>", connectionstyle="arc3,rad=0.0", color=viz.COLORS['primary'], lw=1.5,
                              shrinkA=20, shrinkB=20)
        ax.add_artist(con)

    # Feedback Loops
    con = ConnectionPatch(xyA=(nodes['Eval']['x'], nodes['Eval']['y']), xyB=(nodes['Prep']['x'], nodes['Prep']['y']),
                          coordsA="data", coordsB="data", axesA=ax, axesB=ax,
                          arrowstyle="-|>", connectionstyle="arc3,rad=-0.2", color=viz.COLORS['RedLine'], lw=1.5, linestyle='--',
                          shrinkA=20, shrinkB=20)
    ax.add_artist(con)
    ax.text(8, 4, "Needs Improvement", ha='center', va='center', fontsize=8, color=viz.COLORS['RedLine'], rotation=-25, backgroundcolor='white')

    return ax


def plot_distributed_training(ax=None):
    """Visualizes 'Data Parallel Training Flow'."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    from matplotlib.patches import FancyBboxPatch, Rectangle

    # Input Data
    p = FancyBboxPatch((4.5, 7), 3, 0.8, boxstyle="round,pad=0.1", fc=viz.COLORS['GreenL'], ec=viz.COLORS['GreenLine'], lw=2)
    ax.add_patch(p)
    ax.text(6, 7.4, "Input Data", ha='center', va='center', fontsize=10, fontweight='bold')

    # Split Arrows
    ax.annotate("", xy=(3, 6), xytext=(6, 7), arrowprops=dict(arrowstyle="->", color=viz.COLORS['primary'], lw=1.5))
    ax.annotate("", xy=(9, 6), xytext=(6, 7), arrowprops=dict(arrowstyle="->", color=viz.COLORS['primary'], lw=1.5))

    # GPU 1 Track
    ax.add_patch(Rectangle((1.5, 2.5), 3, 3.5, fill=False, edgecolor=viz.COLORS['BlueLine'], linestyle='--', lw=1))
    ax.text(3, 6.2, "GPU 1", ha='center', fontweight='bold', color=viz.COLORS['BlueLine'])

    ax.text(3, 5.5, "Batch 1", ha='center', fontsize=9, bbox=dict(facecolor='white', edgecolor=viz.COLORS['primary']))
    ax.annotate("", xy=(3, 4.5), xytext=(3, 5.2), arrowprops=dict(arrowstyle="->", color=viz.COLORS['primary']))
    ax.text(3, 4.0, "Forward/\nBackward", ha='center', va='center', fontsize=9, bbox=dict(facecolor=viz.COLORS['BlueL'], edgecolor='none'))
    ax.annotate("", xy=(3, 3.0), xytext=(3, 3.5), arrowprops=dict(arrowstyle="->", color=viz.COLORS['primary']))
    ax.text(3, 2.8, "Gradients", ha='center', fontsize=9, style='italic')

    # GPU 2 Track
    ax.add_patch(Rectangle((7.5, 2.5), 3, 3.5, fill=False, edgecolor=viz.COLORS['BlueLine'], linestyle='--', lw=1))
    ax.text(9, 6.2, "GPU 2", ha='center', fontweight='bold', color=viz.COLORS['BlueLine'])

    ax.text(9, 5.5, "Batch 2", ha='center', fontsize=9, bbox=dict(facecolor='white', edgecolor=viz.COLORS['primary']))
    ax.annotate("", xy=(9, 4.5), xytext=(9, 5.2), arrowprops=dict(arrowstyle="->", color=viz.COLORS['primary']))
    ax.text(9, 4.0, "Forward/\nBackward", ha='center', va='center', fontsize=9, bbox=dict(facecolor=viz.COLORS['BlueL'], edgecolor='none'))
    ax.annotate("", xy=(9, 3.0), xytext=(9, 3.5), arrowprops=dict(arrowstyle="->", color=viz.COLORS['primary']))
    ax.text(9, 2.8, "Gradients", ha='center', fontsize=9, style='italic')

    # Synchronization
    p = FancyBboxPatch((4, 1), 4, 1, boxstyle="round,pad=0.1", fc=viz.COLORS['VioletL'], ec=viz.COLORS['VioletLine'], lw=2)
    ax.add_patch(p)
    ax.text(6, 1.5, "Gradient Aggregation\n(AllReduce)", ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows to Sync
    ax.annotate("", xy=(5, 2), xytext=(3, 2.5), arrowprops=dict(arrowstyle="->", color=viz.COLORS['primary'], lw=1.5))
    ax.annotate("", xy=(7, 2), xytext=(9, 2.5), arrowprops=dict(arrowstyle="->", color=viz.COLORS['primary'], lw=1.5))

    # Update Arrow
    ax.annotate(
        "Model Update",
        xy=(6, 4.0),
        xytext=(6, 2.0),
        arrowprops=dict(arrowstyle="->", color=viz.COLORS['RedLine'], lw=2, linestyle='dashed'),
        ha='center',
        va='center',
        fontsize=9,
        color=viz.COLORS['RedLine'],
        backgroundcolor='white',
    )

    return ax


# Set style
viz.set_book_style()

print("Generating ML Lifecycle...")
plot_ml_lifecycle()
plt.savefig(f"{OUTPUT_DIR}/ml_lifecycle.png")
plt.close('all')

print("Generating Distributed Training...")
plot_distributed_training()
plt.savefig(f"{OUTPUT_DIR}/distributed_training.png")
plt.close('all')

print(f"Plots saved to {OUTPUT_DIR}")
