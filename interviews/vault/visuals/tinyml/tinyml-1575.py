import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 1.5))
ax.broken_barh([(0, 48), (50, 48)], (0, 1), facecolors='#4a90c4')
ax.broken_barh([(48, 2)], (0, 1), facecolors='#c87b2a', label='2s RTO')
ax.set_yticks([])
ax.set_xlabel('Time')
ax.legend()
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')