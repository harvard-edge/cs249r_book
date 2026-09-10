import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 2))
ax.broken_barh([(0, 0.1), (1.33, 0.1)], (0, 12), facecolors='#4a90c4')
ax.set_xlim(0, 2)
ax.set_ylim(0, 15)
ax.set_ylabel('Power (mW)')
ax.set_xlabel('Time (s)')
ax.set_title('Duty Cycle at Max Frequency')
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)