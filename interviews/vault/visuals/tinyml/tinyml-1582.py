import os
import matplotlib.pyplot as plt

labels = ['SRAM Access', 'Flash Access']
cycles = [1, 4]

plt.figure(figsize=(5, 4))
plt.bar(labels, cycles, color=['#4a90c4', '#c87b2a'])
plt.ylabel('Clock Cycles')
plt.title('Memory Fetch Latency')

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')