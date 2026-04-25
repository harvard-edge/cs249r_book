import os
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4,4))
ax.bar(['KV Cache'], [1073], color='#fdebd0', edgecolor='#c87b2a')
ax.axhline(1000, color='red', linestyle='--', label='1GB Strict Limit')
ax.set_ylabel('Memory (MB)')
plt.legend()
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')