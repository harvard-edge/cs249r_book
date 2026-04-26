import os
import matplotlib.pyplot as plt

mem_types = ['SRAM Cap', 'Flash Cap', 'Model Size']
values = [256, 1024, 300]
plt.figure(figsize=(6, 4))
bars = plt.bar(mem_types, values, color=['#4a90c4', '#4a90c4', '#c87b2a'])
plt.axhline(256, color='red', linestyle='--')
plt.ylabel('Size (KB)')
plt.title('Memory Constraints')

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')