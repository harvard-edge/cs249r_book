import os
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 2))
ax.barh('SRAM Arena', 8, color='#4a90c4', edgecolor='black', label='Input (8KB)')
ax.barh('SRAM Arena', 4, left=8, color='#3d9e5a', edgecolor='black', label='Output (4KB)')
ax.set_xlim(0, 15)
ax.set_xlabel('Size (KB)')
ax.legend()

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')