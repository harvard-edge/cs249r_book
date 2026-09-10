import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
labels = ['Polling (100 wakes)', 'DMA Batch (1 wake)']
energy = [50, 12]
ax.bar(labels, energy, color=['#fdebd0', '#d4edda'], edgecolor='#c87b2a')
ax.set_ylabel('Relative Energy per Second')
ax.set_title('Energy: Polling vs DMA Batching')
plt.savefig(os.environ['VISUAL_OUT_PATH'], format='svg', bbox_inches='tight')