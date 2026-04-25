import os
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 5))
stages = ['Network\nCapacity', 'PCIe Gen5\nCapacity', 'Required Load\n(Raw FP16)', 'Required Load\n(JPEG)']
bandwidths = [6.25, 128.0, 32.2, 1.0]
colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']

bars = ax.bar(stages, bandwidths, color=colors)
ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
ax.set_title('Data Pipeline Bottleneck Diagnosis', fontsize=14)
ax.axhline(y=6.25, color='#e74c3c', linestyle='--', zorder=0)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f} GB/s", ha='center', va='bottom', fontsize=11)

ax.set_ylim(0, 150)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = os.environ.get("VISUAL_OUT_PATH", "output.svg")
plt.savefig(out, format="svg", bbox_inches="tight")