import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,2))
ax.broken_barh([(0,5)], (1, 1), facecolors='#fdebd0', label='Cold Start')
ax.broken_barh([(0,1)], (3, 1), facecolors='#d4edda', label='NVMe Reload')
ax.set_yticks([1.5, 3.5])
ax.set_yticklabels(['Without CP', 'With CP'])
ax.set_xlabel('Recovery Time (s)')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')