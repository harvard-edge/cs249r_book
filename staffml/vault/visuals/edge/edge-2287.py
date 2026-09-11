import os
import matplotlib.pyplot as plt

labels = ['LPDDR5 Bandwidth', 'PCIe Bottleneck']
values = [204.8, 0]
plt.figure(figsize=(6, 4))
plt.bar(labels, values, color=['#3d9e5a', '#c87b2a'])
plt.ylabel('Bandwidth (GB/s)')
plt.title('Jetson Unified Memory')

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')