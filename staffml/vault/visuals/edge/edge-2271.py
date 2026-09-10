import os
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4,4))
ax.bar(['Requested'], [150], label='CNN (150)', color='#cfe2f3')
ax.bar(['Requested'], [80], bottom=[150], label='GPU (80)', color='#fdebd0')
ax.axhline(204.8, color='red', linestyle='--', label='204.8 GB/s Limit')
ax.set_ylabel('Bandwidth (GB/s)')
plt.legend()
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')