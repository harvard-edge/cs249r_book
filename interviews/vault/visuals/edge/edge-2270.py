import os
import matplotlib.pyplot as plt

tokens = [1, 512, 1024, 2048]
mem = [1/1024, 0.5, 1.0, 2.0]
plt.figure(figsize=(6,4))
plt.bar([str(t) for t in tokens], mem, color='#cfe2f3', edgecolor='#4a90c4')
plt.xlabel('Sequence Length')
plt.ylabel('KV Cache Size (GB)')
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')