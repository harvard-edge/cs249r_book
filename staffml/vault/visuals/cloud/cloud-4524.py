import os
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(6,2))
ax.barh(['Network'], [50], color='#fdebd0', edgecolor='#c87b2a', label='Latency (ms)')
ax.barh(['Buffer Fill'], [50], color='#d4edda', edgecolor='#3d9e5a', label='100MB Buffer')
ax.set_xlabel('Time (ms)')
ax.legend()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')