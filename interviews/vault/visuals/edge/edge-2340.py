import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 3))
ax.barh(['LPDDR5', 'On-Chip SRAM'], [100, 2000], color=['#c87b2a', '#4a90c4'])
ax.set_xlabel('Bandwidth (GB/s)')
ax.set_title('Tier Bandwidth Profiles')
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)