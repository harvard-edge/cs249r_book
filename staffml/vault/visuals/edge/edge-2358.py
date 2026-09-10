import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3))
ax.fill_between([0, 100, 100, 1000], [15, 15, 5, 5], color='#d4edda', alpha=0.7)
ax.plot([0, 100, 100, 1000], [15, 15, 5, 5], color='#3d9e5a', lw=2)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Power (W)')
ax.set_title('AGX Orin Frame Processing Power')
plt.savefig(os.environ['VISUAL_OUT_PATH'], format='svg', bbox_inches='tight')