import os
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(4,3))
ax.bar(['Capacity', 'Camera'], [204.8, 1.49], color=['#cfe2f3', '#fdebd0'], edgecolor=['#4a90c4', '#c87b2a'])
ax.set_ylabel('Bandwidth (GB/s)')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')