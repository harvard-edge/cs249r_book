import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5,3))
ax.bar(['Disk', 'CPU', 'PCIe', 'GPU'], [8000, 5200, 15000, 5120], color='#cfe2f3', edgecolor='#4a90c4')
ax.axhline(5120, color='#c87b2a', linestyle='--', label='Min Required')
ax.set_ylabel('Images / Second')
ax.legend()
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')