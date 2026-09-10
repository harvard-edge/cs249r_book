import os
import matplotlib.pyplot as plt

labels = ['L1/L2 Cache', 'System RAM']
bw = [1000, 50]
plt.figure(figsize=(6, 4))
plt.bar(labels, bw, color='#4a90c4')
plt.ylabel('Bandwidth (GB/s)')
plt.title('A17 Pro Memory Bandwidth')

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')