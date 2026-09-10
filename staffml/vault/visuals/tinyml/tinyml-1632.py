import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 3))
ax.broken_barh([(0, 10), (12, 10)], (10, 9), facecolors=('tab:blue', 'tab:green'))
ax.vlines(10, 5, 25, colors='r', linestyles='solid', label='Checkpoint')
ax.vlines(11.5, 5, 25, colors='k', linestyles='dashed', label='Brownout')
ax.set_ylim(5, 25)
ax.set_xlim(0, 25)
ax.set_yticks([])
ax.set_xlabel('Time (deciseconds)')
ax.legend()
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)