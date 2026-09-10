import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,2))
ax.plot([0, 5], [0, 100], color='#4a90c4')
ax.plot([5, 5.1], [100, 0], color='#c87b2a', label='CPU Wake/Drain')
ax.plot([5.1, 10], [0, 100], color='#4a90c4')
ax.set_ylabel('Buffer %')
ax.set_xlabel('Time (s)')
ax.legend()
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')