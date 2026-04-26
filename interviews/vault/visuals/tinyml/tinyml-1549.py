import matplotlib.pyplot as plt
import os

phases = ['Reboot/Init', 'Scan Meta', 'Read State']
times = [8.0, 1.0, 2.5]
colors = ['#cfe2f3', '#fdebd0', '#d4edda']

fig, ax = plt.subplots(figsize=(6, 2))
current = 0
for i in range(3):
    ax.barh(0, times[i], left=current, color=colors[i], edgecolor='black', label=phases[i])
    current += times[i]

ax.set_yticks([])
ax.set_xlim(0, 16)
ax.axvline(x=15, color='red', linestyle='--', label='15ms RTO Limit')
ax.set_xlabel('Time (ms)')
ax.set_title('RTO Timeline Profile')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol=4)

out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)