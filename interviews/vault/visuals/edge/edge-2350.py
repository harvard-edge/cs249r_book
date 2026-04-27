import os
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 2))
ax.plot([0, 1, 1, 3, 3, 5], [1, 1, 0, 0, 1, 1], color='#4a90c4', lw=2)
ax.axvspan(1, 3, color='#fdebd0', alpha=0.5, label='2s RTO Window')
ax.text(2, 0.5, 'Reboot & Restore', ha='center')
ax.set_yticks([])
ax.set_xlabel('Time (s)')
ax.legend()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')