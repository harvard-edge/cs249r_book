import os
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(5,2))
ax.barh(['Sequential'], [2], color='#d4edda', edgecolor='#3d9e5a', label='ISP')
ax.barh(['Sequential'], [3], left=[2], color='#cfe2f3', edgecolor='#4a90c4', label='NPU')
ax.legend()
ax.set_xlabel('Latency (ms)')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')