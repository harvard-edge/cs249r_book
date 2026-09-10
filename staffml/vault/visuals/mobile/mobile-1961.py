import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(['L2 BW', 'DRAM BW', 'Effective'], [500, 50, 55], color=['#4a90c4', '#c87b2a', '#3d9e5a'])
ax.set_ylabel('Bandwidth (GB/s)')
ax.set_title('Amdahls Law on Memory BW')
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)