import os
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3))
stages = ['Stage 2', 'Stage 1']
ax.barh(1, 10, left=0, color='#cfe2f3', edgecolor='#4a90c4', label='Frame 1')
ax.barh(0, 10, left=10, color='#cfe2f3', edgecolor='#4a90c4')
ax.barh(1, 10, left=10, color='#d4edda', edgecolor='#3d9e5a', label='Frame 2')
ax.barh(0, 10, left=20, color='#d4edda', edgecolor='#3d9e5a')

ax.set_yticks([0, 1])
ax.set_yticklabels(stages)
ax.set_xlabel('Time (ms)')
ax.legend()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')