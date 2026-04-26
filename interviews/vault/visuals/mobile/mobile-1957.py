import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3))
ax.broken_barh([(0, 0.5), (0.5, 0.5), (1.0, 0.5)], (10, 8), facecolors='#4a90c4', label='Compute Thread')
ax.broken_barh([(0.5, 0.1), (1.5, 0.1)], (0, 8), facecolors='#c87b2a', label='I/O Thread (Write)')
ax.set_ylim(0, 20)
ax.set_yticks([])
ax.set_xlabel('Time (s)')
ax.legend(loc='upper right')
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)