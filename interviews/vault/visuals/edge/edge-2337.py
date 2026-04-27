import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 2))
ax.barh(['Bandwidth'], [60], color='#4a90c4', label='KV Cache (60 GB/s)')
ax.barh(['Bandwidth'], [144.8], left=[60], color='#d4edda', alpha=0.3, label='Headroom')
ax.axvline(204.8, color='red', linestyle='--', label='Orin Max (204.8 GB/s)')
ax.legend(loc='lower right')
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)