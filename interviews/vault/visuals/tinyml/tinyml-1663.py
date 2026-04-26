import os
import matplotlib.pyplot as plt
import numpy as np
t = np.linspace(0, 60, 600)
y = np.where((t > 15)&(t < 17), 12, 0.1)
fig, ax = plt.subplots(figsize=(6,2))
ax.plot(t, y, color='#3d9e5a')
ax.set_ylabel('Power (mW)')
ax.set_xlabel('Time (minutes)')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')