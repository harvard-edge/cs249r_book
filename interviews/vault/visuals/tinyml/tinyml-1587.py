import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,2))
ax.axvline(2, color='#4a90c4', linestyle='-', label='Checkpoint')
ax.axvline(4, color='#4a90c4', linestyle='-')
ax.annotate('Failure', xy=(5, 0), xytext=(5, 1), arrowprops=dict(facecolor='#c87b2a', shrink=0.05))
ax.axvline(6, color='#3d9e5a', linestyle='--', label='Recovery')
ax.set_xlim(0, 8)
ax.set_yticks([])
ax.legend(loc='upper right')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')