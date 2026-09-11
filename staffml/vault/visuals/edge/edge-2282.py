import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,3))
colors = ['#cfe2f3', '#d4edda', '#fdebd0', '#4a90c4']
for i in range(4):
    ax.broken_barh([(i*16, 16)], (40 - i*10, 8), facecolors=colors[i])
ax.set_yticks([14, 24, 34, 44])
ax.set_yticklabels(['Display', 'NPU', 'ISP', 'Sensor'])
ax.set_xlabel('Time (ms)')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')