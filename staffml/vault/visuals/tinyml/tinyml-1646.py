import os
import matplotlib.pyplot as plt

memories = ['SRAM Capacity', 'Model Weights', 'Flash Capacity']
sizes = [256, 500, 1024]
colors = ['#c87b2a', '#4a90c4', '#3d9e5a']

plt.figure(figsize=(6, 3))
plt.bar(memories, sizes, color=colors)
plt.axhline(500, color='gray', linestyle='--')
plt.ylabel('Capacity (KB)')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')