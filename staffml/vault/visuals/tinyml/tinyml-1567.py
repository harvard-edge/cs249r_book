import os
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5,3))
ax.bar(['Flash Read', 'SRAM Write'], [10, 400], color=['#c87b2a', '#4a90c4'])
ax.set_ylabel('Bandwidth (MB/s)')
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')