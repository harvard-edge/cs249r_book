import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,2))
ax.broken_barh([(0,3)], (0,1), facecolors='#cfe2f3')
ax.broken_barh([(3,1)], (0,1), facecolors='#c87b2a', label='Save')
ax.broken_barh([(4,2)], (0,1), facecolors='#d4edda', label='Interrupt')
ax.broken_barh([(6,1)], (0,1), facecolors='#c87b2a', label='Restore')
ax.broken_barh([(7,3)], (0,1), facecolors='#cfe2f3')
ax.set_yticks([])
ax.set_xlabel('Time')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')