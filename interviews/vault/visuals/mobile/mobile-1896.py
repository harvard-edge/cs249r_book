import os
import matplotlib.pyplot as plt
bw = [100, 60]
labels = ['L3 Cache (80% Hit)', 'Main Memory (20% Miss)']
plt.bar(labels, bw, color=['#4a90c4', '#c87b2a'])
plt.ylabel('Bandwidth (GB/s)')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')