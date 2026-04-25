import os
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([0, 5, 5, 5.1, 5.1], [1, 1, 0, 0, 1], color='#4a90c4', lw=2)
ax.axvline(5, color='#c87b2a', linestyle='--', label='Crash Event')
ax.annotate('RPO: 5 min', xy=(2.5, 0.5), ha='center')
ax.annotate('RTO: 10s', xy=(5.05, 0.5), ha='left', rotation=90)
ax.set_yticks([])
ax.set_xlabel('Time (minutes)')
ax.legend()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')