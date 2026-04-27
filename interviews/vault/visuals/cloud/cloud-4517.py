import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
colors = ['#cfe2f3', '#d4edda', '#fdebd0', '#cccccc']
for i in range(4):
    for j in range(4):
        ax.broken_barh([(i + j, 0.8)], (3 - i - 0.4, 0.8), facecolors=colors[j%4], edgecolors='#4a90c4')
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['GPU 3', 'GPU 2', 'GPU 1', 'GPU 0'])
ax.set_xlabel('Time Steps')
ax.set_title('Pipeline Bubble in GPipe (P=4, M=4)')
plt.savefig(os.environ['VISUAL_OUT_PATH'], format='svg', bbox_inches='tight')