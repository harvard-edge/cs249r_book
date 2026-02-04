
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Ensure the 'quarto' directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physx import viz

def plot_technology_s_curve(ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(10, 6))
    
    # Time range: 1980 to 2030
    years = np.linspace(1980, 2030, 500)
    
    # Sigmoid function: L / (1 + exp(-k*(x-x0)))
    def sigmoid(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))
    
    # Curve 1: General Purpose Computing (Moore's Law / Dennard Scaling Era)
    # Starts low, grows fast (1990-2005), saturates (2010-2020)
    # L=100 (normalized peak), k=0.2 (growth rate), x0=2000 (midpoint)
    cpu_curve = sigmoid(years, 100, 0.25, 2000)
    
    # Curve 2: Domain Specific Architectures (Accelerator Era)
    # Starts later (2010), grows fast now (2020+), higher potential ceiling
    # L=10000 (100x potential), k=0.3 (faster growth), x0=2020 (midpoint)
    # We offset it so it starts visible around 2012
    accel_curve = sigmoid(years, 10000, 0.35, 2022)
    
    # Plotting
    ax.plot(years, cpu_curve, color=viz.COLORS['grid'], linewidth=3, label='General Purpose (CPU)')
    ax.plot(years, accel_curve, color=viz.COLORS['BlueLine'], linewidth=3, label='Domain Specific (Accelerator)')
    
    # Fill the gap (The Shift)
    mask = (years > 2012) & (accel_curve > cpu_curve)
    ax.fill_between(years, cpu_curve, accel_curve, where=mask, color=viz.COLORS['BlueL'], alpha=0.2)
    
    # Log scale to show the magnitude difference
    ax.set_yscale('log')
    ax.set_ylim(0.1, 20000)
    ax.set_xlim(1980, 2030)
    
    # Annotations
    
    # Phase 1: Ferment/Take-off for CPU
    ax.text(1990, 2, "Moore's Law\n(Exponential Growth)", color=viz.COLORS['primary'], ha='center', fontsize=9, rotation=25, alpha=0.6)
    
    # Phase 2: Saturation for CPU
    ax.text(2016, 150, "Dennard Scaling Ends\n(Saturation)", color=viz.COLORS['primary'], ha='center', fontweight='bold', fontsize=9)
    
    # Phase 3: The Shift
    ax.annotate("The Paradigm Shift\n(Hardware-Software Co-design)", 
                xy=(2016, 50), xytext=(2005, 0.5),
                arrowprops=dict(facecolor=viz.COLORS['RedLine'], arrowstyle='->', lw=2, color=viz.COLORS['RedLine']),
                fontsize=10, fontweight='bold', color=viz.COLORS['RedLine'], 
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2))
                
    # Phase 4: Take-off for Accelerators
    ax.text(2025, 1000, "Era of Accelerators\n(Matrix Math focus)", color=viz.COLORS['BlueLine'], ha='center', fontweight='bold', fontsize=9, rotation=35)
    
    # Phase 5: The Gap (Why we do this)
    ax.annotate("", xy=(2029, 9000), xytext=(2029, 105),
                arrowprops=dict(arrowstyle="<->", color=viz.COLORS['primary'], lw=1.5))
    ax.text(2028.5, 900, "The Systems Gap\n(~100x)", ha='right', va='center', fontsize=9, fontweight='bold', color=viz.COLORS['primary'])
    
    # Axis labels
    ax.set_xlabel('Year')
    ax.set_ylabel('Performance / Efficiency (Log Scale)')
    
    # Legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Title (Optional, usually caption handles it)
    # ax.set_title("The Technology S-Curve: From CPU to Accelerator", fontsize=12, fontweight='bold')
    
    return ax

if __name__ == "__main__":
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "images", "generated")
    os.makedirs(output_dir, exist_ok=True)
    
    viz.set_book_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_technology_s_curve(ax)
    
    output_path = os.path.join(output_dir, "technology_s_curve.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Generated S-curve plot at: {output_path}")
