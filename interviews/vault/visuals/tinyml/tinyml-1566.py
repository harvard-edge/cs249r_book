import os
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 2))
ax.barh(['Ideal Active'], [10], left=[50], color='#4a90c4')
ax.barh(['Drift Guardband'], [60], left=[0], color='#c87b2a')
ax.set_xlabel('Time (ms)')
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')