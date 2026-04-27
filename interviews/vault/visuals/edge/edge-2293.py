import os
import matplotlib.pyplot as plt

phases = ['Crash', 'Rebooting', 'Reloading State', 'Active']
times = [0, 5, 2, 3]
colors = ['white', '#c87b2a', '#fdebd0', '#3d9e5a']

fig, ax = plt.subplots(figsize=(6, 2))
left = 0
for i in range(1, 4):
    ax.barh('System Status', times[i], left=left, color=colors[i], label=phases[i])
    left += times[i]
ax.set_xlabel('Time (s)')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.7), ncol=3)

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')