import os
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 5))
stages = ['Raw Data\n(5 GB/s)', 'Ring Buffer', 'A17 NE Filter', 'NVMe Storage']
volumes = [144, 144, 16, 16]
colors = ['#cfe2f3', '#cfe2f3', '#fdebd0', '#d4edda']
edges = ['#4a90c4', '#4a90c4', '#c87b2a', '#3d9e5a']

ax.bar(stages, volumes, color=colors, edgecolor=edges, linewidth=2)
ax.set_ylabel('Total Data Volume per Shift (TB)')
ax.set_title('Edge Pruning Pipeline Throughput')
for i, v in enumerate(volumes):
    ax.text(i, v + 2, f'{v} TB', ha='center', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.5)
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)