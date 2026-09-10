import matplotlib.pyplot as plt
import os

labels = ['Recompute', 'Checkpoint + Load']
times = [15, 8]
colors = ['#c87b2a', '#3d9e5a']

fig, ax = plt.subplots(figsize=(6, 2))
ax.barh(labels, times, color=colors, edgecolor='black')
ax.set_xlabel('Time (ms)')
ax.set_title('Recovery Strategy Cost')
ax.axvline(x=8, color='grey', linestyle='--')

out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)