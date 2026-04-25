import os
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.barh(['ISP', 'Mem Copy', 'NPU', 'Display'], [2, 3, 10, 5], left=[0, 2, 5, 15], color='#cfe2f3', edgecolor='#4a90c4')
ax.set_xlabel('Time (ms)')
ax.set_title('Pipeline Stages')
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')