import os
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4, 5))
ax.bar(['Session 1'], [3.5], label='INT4 Weights (3.5GB)', color='#cfe2f3')
ax.bar(['Session 1'], [2.0], bottom=[3.5], label='FP16 KV Cache (2.0GB)', color='#d4edda')
ax.set_ylabel('Memory (GB)')
ax.legend()
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)