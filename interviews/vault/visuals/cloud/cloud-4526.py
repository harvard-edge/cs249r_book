import os
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(5,2))
ax.barh(['vRAM'], [80], color='#cfe2f3', edgecolor='#4a90c4', label='H100 Capacity')
ax.barh(['KV Demand'], [85.9], color='#fdebd0', edgecolor='#c87b2a', label='KV Cache')
ax.set_xlabel('Gigabytes (GB)')
ax.legend(loc='lower right')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')