import os
import matplotlib.pyplot as plt
import numpy as np
tiers = ['SRAM (256KB)', 'Flash (1MB)']
activations = [50, 0]
weights = [206, 94]
unused = [0, 930]
fig, ax = plt.subplots(figsize=(8, 3))
ax.barh(tiers, activations, label='Activations (50KB)', color='#fdebd0')
ax.barh(tiers, weights, left=activations, label='Weights', color='#cfe2f3')
ax.barh(tiers, unused, left=np.add(activations, weights), label='Unused', color='lightgray')
ax.set_xlabel('Capacity (KB)')
ax.set_title('Memory Tier Allocation (Cortex-M4)')
ax.legend()
plt.tight_layout()
out = os.environ.get('VISUAL_OUT_PATH', 'plot.svg')
plt.savefig(out, format='svg', bbox_inches='tight')