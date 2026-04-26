import os
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3))
ax.barh('Frame 1', 15, left=0, color='#cfe2f3', edgecolor='#4a90c4', label='CPU (15ms)')
ax.barh('Frame 1', 10, left=15, color='#d4edda', edgecolor='#3d9e5a', label='NPU (10ms)')
ax.barh('Frame 2', 15, left=25, color='#cfe2f3', edgecolor='#4a90c4')
ax.barh('Frame 2', 10, left=40, color='#d4edda', edgecolor='#3d9e5a')
ax.set_xlabel('Time (ms)')
ax.legend()

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')