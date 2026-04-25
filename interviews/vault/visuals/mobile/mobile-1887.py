import os
import matplotlib.pyplot as plt

ctx = [1000, 2000, 3814, 5000]
mem = [128, 256, 500, 640]
plt.figure(figsize=(6,4))
plt.bar([str(c) for c in ctx], mem, color='#cfe2f3', edgecolor='#4a90c4')
plt.axhline(500, color='red', linestyle='--', label='500MB Limit')
plt.xlabel('Sequence Length')
plt.ylabel('Memory (MB)')
plt.legend()
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')