import matplotlib.pyplot as plt
import os

labels = ['Write temp.pt', 'fsync()', 'rename()']
starts = [0, 45, 48]
durations = [45, 3, 1]
colors = ['#cfe2f3', '#fdebd0', '#d4edda']

fig, ax = plt.subplots(figsize=(6, 2))
for i in range(3):
    ax.barh(0, durations[i], left=starts[i], color=colors[i], edgecolor='black', label=labels[i])

ax.set_yticks([])
ax.set_xlim(0, 55)
ax.axvline(x=50, color='red', linestyle='--', label='OS SIGKILL (50ms)')
ax.set_xlabel('Time (ms)')
ax.set_title('Atomic Checkpoint Sequence')
ax.legend(loc='lower left', bbox_to_anchor=(0, -0.6), ncol=4)

out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)