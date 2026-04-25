import os
import matplotlib.pyplot as plt

labels = ['x1 Lane', 'x4 Lanes']
bw = [0.985, 3.94]

plt.figure(figsize=(4, 4))
plt.bar(labels, bw, color='#4a90c4')
plt.ylabel('Bandwidth (GB/s)')
plt.title('PCIe Gen 3 Link')

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')