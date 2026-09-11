import os
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 2))
ax.barh(['CPU Crop', 'GPU Filter', 'NPU Detect'], [5, 10, 15], left=[0, 5, 15], color='#cfe2f3', edgecolor='#4a90c4')
ax.set_xlabel('Time (ms)')
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')