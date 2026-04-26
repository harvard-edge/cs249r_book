import matplotlib.pyplot as plt
import os

labels = ['Naive (Memcpy)', 'Zero-Copy (Unified)']
bw = [12.0, 6.0]
colors = ['#c87b2a', '#3d9e5a']

fig, ax = plt.subplots(figsize=(6, 2))
ax.barh(labels, bw, color=colors, edgecolor='black')
ax.set_xlabel('Unified Memory Bandwidth (GB/s)')
ax.set_title('Pipeline Bandwidth Optimization')

out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)