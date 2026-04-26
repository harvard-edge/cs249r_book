import os
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 2))
ax.scatter([0, 60], [1, 1], color='#3d9e5a', s=100, label='Checkpoint')
ax.axvspan(0, 59, color='#fdebd0', label='Unsaved Progress')
ax.axvline(59, color='#c87b2a', linestyle='--', label='Crash')
ax.set_yticks([])
ax.set_xlabel('Time (minutes)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')