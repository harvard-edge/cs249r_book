import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,3))
ax.broken_barh([(0, 1), (10, 1), (20, 1)], (0, 1), facecolors='#cfe2f3', edgecolors='#4a90c4')
ax.annotate('Max Data Loss (RPO)', xy=(10, 0.5), xytext=(3, 1.2), arrowprops=dict(facecolor='black', arrowstyle='<->'))
ax.set_yticks([])
ax.set_xlabel('Time')
ax.set_title('Checkpoint Interval Dictating RPO')
plt.savefig(os.environ['VISUAL_OUT_PATH'], format='svg', bbox_inches='tight')