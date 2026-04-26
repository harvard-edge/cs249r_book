import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,2))
ax.plot([0, 4], [0, 0], color='#4a90c4', linewidth=2, label='Deep Sleep')
ax.plot([4, 4], [0, 10], color='#c87b2a', linestyle='--')
ax.plot([4, 6], [10, 10], color='#3d9e5a', linewidth=2, label='Active Infer')
ax.set_yticks([])
ax.set_xlabel('Time')
ax.legend()
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')