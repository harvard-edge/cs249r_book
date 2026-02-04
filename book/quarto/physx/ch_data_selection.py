import matplotlib.pyplot as plt
import numpy as np
import viz
import os

def plot_data_selection_limits(ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(8, 5))

    # --- Data ---
    # Years for the trend line
    years_trend = np.linspace(2012, 2028, 100)
    
    # Model Data Points (Approximate)
    models = [
        {'Name': 'GPT-2', 'Year': 2019, 'Tokens': 1.5e10, 'Offset': (-25, 5)},
        {'Name': 'GPT-3', 'Year': 2020, 'Tokens': 3e11, 'Offset': (-25, 5)},
        {'Name': 'Chinchilla', 'Year': 2022, 'Tokens': 1.4e12, 'Offset': (35, -10)},
        {'Name': 'Llama 2', 'Year': 2023, 'Tokens': 2e12, 'Offset': (35, -5)},
        {'Name': 'Llama 3', 'Year': 2024, 'Tokens': 1.5e13, 'Offset': (-30, 15)},
    ]
    
    # Trend Line Calculation (Exponential Growth)
    # Fit roughly to GPT-2 (2019, 1.5e10) and Llama 3 (2024, 1.5e13)
    slope = 0.635
    intercept = 10.176 - slope * 2019 
    trend_tokens = 10**(slope * years_trend + intercept)
    
    # --- Plotting ---
    
    # 1. High Quality Text Stock (The Limit)
    limit_low = 1e13
    limit_high = 1e14
    
    ax.fill_between(years_trend, limit_low, limit_high, color=viz.COLORS['OrangeL'], alpha=0.4, label='High-Quality Text Stock')
    ax.text(2013, 2.5e13, "High-Quality Public Text Stock\n(Books, Papers, Code, Web)", 
            color=viz.COLORS['OrangeLine'], fontweight='bold', fontsize=10, va='center')

    # 2. Consumption Trend
    ax.plot(years_trend, trend_tokens, color=viz.COLORS['BlueLine'], linewidth=2.5, label='Training Data Demand', zorder=4)
    
    # 3. Model Markers
    for m in models:
        ax.scatter(m['Year'], m['Tokens'], color=viz.COLORS['BlueLine'], s=70, zorder=5, edgecolors='white', linewidth=1.5)
        
        ax.annotate(m['Name'], (m['Year'], m['Tokens']), xytext=m['Offset'], textcoords='offset points',
                    fontsize=9, fontweight='bold', color=viz.COLORS['BlueLine'], ha='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))

    # --- Formatting ---
    ax.set_yscale('log')
    ax.set_ylim(1e9, 1e15) # 1B to 1PB
    ax.set_xlim(2012, 2028)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Dataset Size (Tokens)')
    ax.grid(True, which="both", ls="-", alpha=0.05)
    
    # Intersection Point Calculation for Annotation
    cross_year = (np.log10(limit_low) - intercept) / slope
    
    ax.annotate("Public Data Exhaustion", 
                xy=(cross_year, limit_low), xytext=(2017, 2e10),
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-0.2", color=viz.COLORS['RedLine'], lw=1.5),
                color=viz.COLORS['RedLine'], fontweight='bold', fontsize=10, ha='center',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1))

    return ax

if __name__ == "__main__":
    viz.set_book_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data_selection_limits(ax)
    
    # Save path relative to this script: ../contents/vol1/data_selection/images/png/running_out_of_data.png
    output_path = os.path.join(os.path.dirname(__file__), '../contents/vol1/data_selection/images/png/running_out_of_data.png')
    output_path = os.path.normpath(output_path)
    
    print(f"Saving to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Done.")
