
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add book/quarto/mlsys to path to import viz
# Script is in book/tools/scripts/genai/
# We need to reach book/quarto/mlsys
# ../../../quarto/mlsys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../quarto/mlsys")))

try:
    import viz
    viz.set_book_style()
    COLORS = viz.COLORS
except ImportError:
    print("Warning: Could not import viz.py, using fallback style.")
    COLORS = {
        "primary": "#333333",
        "RedLine": "#CB202D",
        "BlueLine": "#006395",
        "GreenLine": "#008F45",
        "OrangeLine": "#E67817",
        "grid": "#CCCCCC"
    }
    plt.style.use('seaborn-v0_8-whitegrid')

# Custom Yellow that fits the palette better than pure yellow but is distinct from Orange
COLORS["YellowLine"] = "#F4D03F" 

def draw_node(ax, x, y, label=None, radius=0.15, color="#E0E0E0"):
    """Draws a circular node."""
    circle = patches.Circle((x, y), radius, facecolor=color, edgecolor=COLORS["primary"], linewidth=1.5, zorder=10)
    ax.add_patch(circle)
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=14, fontweight='bold', zorder=11, color=COLORS["primary"])
    return circle

def draw_rect(ax, x, y, width, height, color):
    """Draws a data rectangle centered at (x, y)."""
    rect = patches.Rectangle((x - width/2, y - height/2), width, height, 
                             facecolor=color, edgecolor=None, zorder=15)
    ax.add_patch(rect)

def draw_arrow(ax, start, end):
    """Draws an arrow from start to end."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=COLORS["primary"], lw=1.5, shrinkA=0, shrinkB=0),
                zorder=5)

def setup_subplot(ax, title):
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.8)
    ax.axis('off')
    # Title at the bottom
    ax.text(0, -0.15, title, ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS["primary"])

def generate_diagram():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Node positions
    src_y = 0.2
    dest_y = 1.4
    
    # Horizontal spacing for 4 nodes
    xs = np.linspace(-0.9, 0.9, 4)
    
    # Colors
    c_red = COLORS["RedLine"]
    c_yellow = COLORS["YellowLine"]
    c_green = COLORS["GreenLine"]
    c_blue = COLORS["BlueLine"]
    palette = [c_red, c_yellow, c_green, c_blue]
    
    # --- 1. Broadcast (Top Left) ---
    ax = axes[0, 0]
    setup_subplot(ax, "Broadcast")
    
    # Source (Bottom)
    draw_node(ax, 0, src_y)
    draw_rect(ax, 0, src_y, 0.08, 0.15, c_red)
    
    # Dests (Top)
    for x in xs:
        draw_node(ax, x, dest_y)
        draw_rect(ax, x, dest_y, 0.08, 0.15, c_red)
        # Arrow
        draw_arrow(ax, (0, src_y + 0.15), (x, dest_y - 0.15))

    # --- 2. Scatter (Top Right) ---
    ax = axes[0, 1]
    setup_subplot(ax, "Scatter")
    
    # Source (Bottom)
    draw_node(ax, 0, src_y)
    # Composite block at source
    w = 0.06
    h = 0.12
    total_w = 4 * w
    start_x = -total_w / 2 + w/2
    for i, color in enumerate(palette):
        draw_rect(ax, start_x + i*w, src_y, w, h, color)
        
    # Dests (Top)
    for i, x in enumerate(xs):
        draw_node(ax, x, dest_y)
        draw_rect(ax, x, dest_y, 0.08, 0.15, palette[i])
        # Arrow
        draw_arrow(ax, (0, src_y + 0.15), (x, dest_y - 0.15))

    # --- 3. Gather (Bottom Left) ---
    ax = axes[1, 0]
    setup_subplot(ax, "Gather")
    
    # Dest (Bottom)
    draw_node(ax, 0, src_y)
    # Composite block at dest
    start_x = -total_w / 2 + w/2
    for i, color in enumerate(palette):
        draw_rect(ax, start_x + i*w, src_y, w, h, color)
        
    # Sources (Top)
    for i, x in enumerate(xs):
        draw_node(ax, x, dest_y)
        draw_rect(ax, x, dest_y, 0.08, 0.15, palette[i])
        # Arrow
        draw_arrow(ax, (x, dest_y - 0.15), (0, src_y + 0.15))

    # --- 4. Reduction (Bottom Right) ---
    ax = axes[1, 1]
    setup_subplot(ax, "Reduction")
    
    # Dest (Bottom)
    draw_node(ax, 0, src_y, label="16")
    
    # Sources (Top)
    values = ["1", "3", "5", "7"]
    for i, x in enumerate(xs):
        draw_node(ax, x, dest_y, label=values[i])
        # Arrow
        draw_arrow(ax, (x, dest_y - 0.15), (0, src_y + 0.15))

    plt.tight_layout()
    
    # Save
    output_path = os.path.abspath("comm_primitives.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Generated image at: {output_path}")

if __name__ == "__main__":
    generate_diagram()
