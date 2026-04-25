import os
import matplotlib.pyplot as plt

stages = ['S3 Read\n(5 GB/s)', 'CPU Decode\n(50 Cores)', 'PCIe Gen5\n(30 GB/s)', 'GPU Process\n(8x H100)']
capacity = [100, 100, 600, 400]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(stages, capacity, color='#4a90c4')
ax.axhline(y=100, color='red', linestyle='--', label='Target Throughput (10k img/s)')
ax.set_ylabel('Throughput Capacity (%)')
ax.set_title('Pipeline Bottleneck Analysis')
ax.legend()
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')