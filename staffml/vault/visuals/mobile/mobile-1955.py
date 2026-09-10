import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 2))
ax.broken_barh([(0, 0.45)], (1, 1), facecolors='#3d9e5a')
ax.axvline(1.0, color='r', linestyle='--', label='OS Kill Signal')
ax.set_xlim(0, 1.2)
ax.set_yticks([])
ax.set_xlabel('Time (s) after preemption warning')
ax.legend()
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)