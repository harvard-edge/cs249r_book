import os
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(6,2))
ax.barh([1], [10], left=[0], color='#cfe2f3', edgecolor='#4a90c4')
ax.barh([0], [15], left=[10], color='#d4edda', edgecolor='#3d9e5a')
ax.set_yticks([0, 1])
ax.set_yticklabels(['Recognition', 'Detection'])
ax.set_xlabel('Time (ms)')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')