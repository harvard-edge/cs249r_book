import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,1.5))
ax.hlines(1, 0, 15, colors='#4a90c4', linewidth=2)
ax.vlines([0, 5, 10, 15], 0.8, 1.2, colors='#c87b2a', linewidth=3)
ax.text(2.5, 1.3, 'RPO = 5 Min', ha='center')
ax.set_xticks([0, 5, 10, 15])
ax.set_xlabel('Time (Minutes)')
ax.set_yticks([])
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')