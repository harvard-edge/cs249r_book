
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys

from . import viz

def plot_broadcasting_visual(ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(10, 4))
    
    # Hide axes
    ax.axis('off')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4)
    
    # Colors
    c_active = viz.COLORS['BlueLine'] # Dark Blue
    c_ghost = viz.COLORS['BlueFill']  # Light Blue
    c_text_active = 'white'
    c_text_ghost = viz.COLORS['BlueLine'] # Dark text on light BG
    
    # Helper to draw a box
    def draw_box(x, y, label, is_ghost=False):
        fc = c_ghost if is_ghost else c_active
        tc = c_text_ghost if is_ghost else c_text_active
        ec = c_active if is_ghost else c_active
        ls = '--' if is_ghost else '-'
        lw = 1
        
        rect = patches.Rectangle((x, y), 1, 1, linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls)
        ax.add_patch(rect)
        ax.text(x + 0.5, y + 0.5, label, ha='center', va='center', color=tc, fontweight='bold', fontsize=10)

    # Tensor A: (3, 1) -> Column Vector
    # Position: x=1
    ax.text(1.5, 3.5, "Tensor A: (3, 1)", ha='center', fontsize=12, fontweight='bold')
    draw_box(1, 2, "A0")
    draw_box(1, 1, "A1")
    draw_box(1, 0, "A2")
    
    # Plus Sign
    ax.text(3, 1.5, "+", fontsize=20, ha='center', va='center')
    
    # Tensor B: (1, 4) -> Row Vector
    # Position: x=4
    ax.text(6, 3.5, "Tensor B: (1, 4)", ha='center', fontsize=12, fontweight='bold')
    draw_box(4, 1, "B0")
    draw_box(5, 1, "B1")
    draw_box(6, 1, "B2")
    draw_box(7, 1, "B3")
    
    # Equals Sign
    ax.text(9, 1.5, "=", fontsize=20, ha='center', va='center')
    
    # Result: (3, 4)
    # Position: x=10
    ax.text(12, 3.5, "Result: (3, 4)", ha='center', fontsize=12, fontweight='bold')
    
    # Row 0 (A0 + Bx)
    draw_box(10, 2, "A0+B0", is_ghost=True)
    draw_box(11, 2, "A0+B1", is_ghost=True)
    draw_box(12, 2, "A0+B2", is_ghost=True)
    draw_box(13, 2, "A0+B3", is_ghost=True)
    # Highlight the "real" A0 source? No, conceptual mix.
    # Actually, usually broadcasting shows the source "stretching".
    # Let's match the TikZ concept: 
    # A is (3,1), B is (1,4).
    # Result (3,4) combines them.
    
    # To match the TikZ "ghost" visual:
    # A becomes (3,4) by replicating columns.
    # B becomes (3,4) by replicating rows.
    # We show the final result grid.
    
    labels = [
        ["A0", "A0", "A0", "A0"],
        ["A1", "A1", "A1", "A1"],
        ["A2", "A2", "A2", "A2"]
    ]
    
    # Draw the grid
    for r in range(3):
        for c in range(4):
            # y goes 2, 1, 0
            y = 2 - r
            x = 10 + c
            
            # Logic: In broadcasting, A is copied across columns, B across rows.
            # Visualizing just the result sum is often cleaner.
            label = f"A{r}+B{c}"
            
            # Use Active color for the "original" positions?
            # A (3,1) implies col 0 is "original" A? No, A is column vector.
            # B (1,4) implies row 0 is "original" B?
            
            # Let's stick to the visual style: Light Blue background for all result cells
            # because they are all computed.
            draw_box(x, y, label, is_ghost=True)
            
    # Draw overlay of "Original" data? 
    # The TikZ diagram had "Active" and "Ghost". 
    # A (3,1) -> Active were (0,0), (1,0), (2,0). Ghosts were the rest.
    # B (1,4) -> Active were (0,0), (0,1), (0,2), (0,3). Ghosts were the rest.
    # The result is the sum.
    
    # Let's just visualize the inputs and output clearly.
    # Inputs are SOLID (Active).
    # Output is LIGHT (Ghost/Result) to distinguish.
    
    return ax

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "images", "generated")
    os.makedirs(output_dir, exist_ok=True)
    
    viz.set_book_style()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_broadcasting_visual(ax)
    
    output_path = os.path.join(output_dir, "broadcasting_python.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Generated Broadcasting plot at: {output_path}")
