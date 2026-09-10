import os
import matplotlib.pyplot as plt

batch_sizes = [1, 5, 11, 12]
memory_gb = [2.14 * b for b in batch_sizes]

plt.figure(figsize=(6, 3))
plt.bar(batch_sizes, memory_gb, color='#cfe2f3', edgecolor='#4a90c4')
plt.axhline(24, color='#c87b2a', linestyle='--', label='24GB Limit')
plt.xlabel('Batch Size')
plt.ylabel('KV Cache Size (GB)')
plt.legend()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')