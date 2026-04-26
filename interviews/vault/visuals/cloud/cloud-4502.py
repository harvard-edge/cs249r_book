import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5,3))
ax.barh(['Physical Block 3', 'Physical Block 1'], [10, 10], color='#cfe2f3', edgecolor='#4a90c4')
ax.set_title('Non-Contiguous Physical Memory Allocation')
ax.set_xlabel('Token Capacity')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')