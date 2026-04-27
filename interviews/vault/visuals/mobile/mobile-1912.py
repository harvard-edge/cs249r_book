import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5,3))
ax.bar(['FP16', 'INT4'], [40, 10], color='#cfe2f3', edgecolor='#4a90c4')
ax.axhline(16, color='red', linestyle='--', label='SRAM Limit')
ax.set_ylabel('Model Size (MB)')
ax.legend()
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')