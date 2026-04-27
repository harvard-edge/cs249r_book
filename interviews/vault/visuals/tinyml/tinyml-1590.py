import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,2))
ax.plot([0, 10], [1, 1], color='#c87b2a', label='RTC Active')
ax.vlines(x=[2, 5, 8], ymin=1, ymax=10, color='#4a90c4', linewidth=3, label='CPU Wake')
ax.set_yticks([])
ax.set_xlabel('Time')
ax.legend(loc='upper right')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')