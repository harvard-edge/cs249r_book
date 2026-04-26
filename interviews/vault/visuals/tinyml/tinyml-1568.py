import os
import matplotlib.pyplot as plt

plt.figure(figsize=(8,3))
plt.plot([0, 5, 5, 6, 10], [1, 1, 0, 1, 1], color='#4a90c4', linewidth=2)
plt.axvline(x=5, color='#c87b2a', linestyle='--', label='Failure (RPO=5s)')
plt.axvline(x=6, color='#3d9e5a', linestyle='--', label='Recovery (RTO=1s)')
plt.fill_between([0, 5], 0, 1, color='#cfe2f3', alpha=0.5, label='Lost Data Window')
plt.yticks([0, 1], ['Down', 'Up'])
plt.xlabel('Time (s)')
plt.legend()
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')