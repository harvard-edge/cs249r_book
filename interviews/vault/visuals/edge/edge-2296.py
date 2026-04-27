import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,2))
ax.broken_barh([(0,1)], (0,1), facecolors='#c87b2a', label='Reset (0.5s)')
ax.broken_barh([(1,4)], (0,1), facecolors='#cfe2f3', label='Reload (2.0s)')
ax.set_xlim(0, 6)
ax.set_yticks([])
ax.legend(loc='upper right')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')