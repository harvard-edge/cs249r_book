import os
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(6,2))
ax.barh(['Memory'], [1.5], color='#d4edda', edgecolor='#3d9e5a', label='Weights (1.5GB)')
ax.barh(['Memory'], [2.5], left=[1.5], color='#cfe2f3', edgecolor='#4a90c4', label='KV Cache (2.5GB)')
ax.set_xlim(0, 4.5)
ax.axvline(4.0, color='red', linestyle='--', label='OS Limit (4GB)')
ax.set_xlabel('Gigabytes (GB)')
ax.legend(loc='lower left')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')