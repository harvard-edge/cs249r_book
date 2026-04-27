import os
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 2))
ax.barh(['Grace Period (1.5s)'], [1500], color='#d4edda', edgecolor='#3d9e5a')
ax.barh(['Grace Period (1.5s)'], [75], color='#c87b2a', edgecolor='black', label='Save (75ms)')
ax.set_xlabel('Time (ms)')
plt.legend()
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')