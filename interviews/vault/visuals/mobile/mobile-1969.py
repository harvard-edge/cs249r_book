import os
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(6, 3))
ax.barh(1, 10, left=0, color='#fdebd0', edgecolor='gray', label='Wasted Space')
ax.barh(1, 3, left=0, color='#4a90c4', edgecolor='black', label='Used Space')

for i in range(4):
    ax.barh(0, 0.8, left=i, color='#4a90c4', edgecolor='black')

ax.set_yticks([0, 1])
ax.set_yticklabels(['PagedAttention', 'Contiguous'])
ax.set_xlim(0, 10)
ax.legend()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')