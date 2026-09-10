import os
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(5,2))
ax.barh(['Pipeline'], [0.2], color='#fdebd0', edgecolor='#c87b2a', label='RAM Copy (0.2ms)')
ax.barh(['Pipeline'], [2.0], left=[0.2], color='#cfe2f3', edgecolor='#4a90c4', label='Inference (2ms)')
ax.legend()
ax.set_xlabel('Latency (ms)')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')