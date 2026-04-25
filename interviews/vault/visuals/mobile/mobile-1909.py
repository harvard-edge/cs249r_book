import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,2))
ax.broken_barh([(0,13), (20,13)], (0, 5), facecolors='#cfe2f3', label='Sleep')
ax.broken_barh([(13,2), (33,2)], (0, 200), facecolors='#fdebd0', label='Transient')
ax.broken_barh([(15,5), (35,5)], (0, 450), facecolors='#d4edda', label='Active')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Power (mW)')
ax.legend(loc='upper right')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')