import os
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(5,3))
stages = ['CPU', 'PCIe', 'GPU']
rates = [1, 64, 15]
ax.bar(stages, rates, color=['#fdebd0', '#d4edda', '#cfe2f3'], edgecolor=['#c87b2a', '#3d9e5a', '#4a90c4'])
ax.set_ylabel('Throughput (GB/s)')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')