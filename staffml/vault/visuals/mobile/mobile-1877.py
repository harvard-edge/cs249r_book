import matplotlib.pyplot as plt
import os

labels = ['LPDDR5 Transfer', 'NPU Compute (Ideal)']
times = [0.200, 0.050]
colors = ['#c87b2a', '#4a90c4']

fig, ax = plt.subplots(figsize=(6, 2))
ax.barh(labels, times, color=colors, edgecolor='black')
ax.set_xlabel('Time (ms)')
ax.set_title('Layer Execution Bottleneck')
ax.axvline(x=0.200, color='red', linestyle='--', label='Memory Wall')

out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)