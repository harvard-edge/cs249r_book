import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,2))
ax.broken_barh([(0,2)], (0,1), facecolors='#cfe2f3', label='ISP Write')
ax.broken_barh([(2,2)], (0,1), facecolors='#d4edda', label='GPU Convert')
ax.broken_barh([(4,3)], (0,1), facecolors='#fdebd0', label='NPU Infer')
ax.set_xlim(0, 8)
ax.set_yticks([])
ax.set_xlabel('Time Pipeline (ms)')
ax.legend(loc='upper right')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')