import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(['Base (50ms@20mW)', 'Fast (25ms@45mW)'], [1.0, 1.125], color=['#4a90c4', '#c87b2a'])
ax.set_ylabel('Active Energy (mJ)')
ax.set_title('DVFS Energy Cost')
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)