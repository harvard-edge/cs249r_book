import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3))
stages = ['NVMe Storage', 'PCIe Gen5 Bus', 'HBM3 Memory']
throughput = [0.75, 1.5, 3350]
ax.barh(stages, throughput, color='#cfe2f3', edgecolor='#4a90c4')
ax.set_xscale('log')
ax.set_xlabel('Required / Max Throughput (GB/s, Log Scale)')
ax.set_title('Pipeline Bandwidth Scaling')
plt.savefig(os.environ['VISUAL_OUT_PATH'], format='svg', bbox_inches='tight')