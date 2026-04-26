import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5,2))
ax.barh(['NPU SRAM', 'LPDDR5 RAM'], [800, 50], color='#cfe2f3', edgecolor='#4a90c4')
ax.set_xlabel('Effective Bandwidth (GB/s)')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')