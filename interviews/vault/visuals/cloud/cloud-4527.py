import os
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(5,3))
ax.bar(['Theoretical HBM', 'Effective'], [5300, 331], color=['#cfe2f3', '#fdebd0'], edgecolor=['#4a90c4', '#c87b2a'])
ax.set_ylabel('Bandwidth (GB/s)')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')