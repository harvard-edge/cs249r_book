import matplotlib.pyplot as plt
import os

labels = ['INT4 Weights', 'KV Cache Capacity']
sizes = [1536, 512]
colors = ['#cfe2f3', '#c87b2a']

fig, ax = plt.subplots(figsize=(6, 2))
ax.barh(0, sizes[0], color=colors[0], edgecolor='black', label=labels[0])
ax.barh(0, sizes[1], left=sizes[0], color=colors[1], edgecolor='black', label=labels[1])
ax.set_yticks([])
ax.set_xlim(0, 2100)
ax.axvline(x=2048, color='red', linestyle='--', label='2048 MB iOS Limit')
ax.set_xlabel('Memory (MB)')
ax.set_title('Memory Distribution for 3B Model')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol=3)

out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)