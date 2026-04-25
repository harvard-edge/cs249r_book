import os
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(6,2))
ax.barh(['Recovery'], [2.0], color='#fdebd0', edgecolor='#c87b2a', label='SSD Read')
ax.set_xlabel('Time (s)')
ax.legend()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')