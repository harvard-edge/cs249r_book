import os
import matplotlib.pyplot as plt
import numpy as np
t = np.linspace(0, 10, 100)
y = np.where((t > 2)&(t < 3) | (t > 7)&(t < 8), 20.5, 0.5)
fig, ax = plt.subplots(figsize=(5,3))
ax.plot(t, y, color='#3d9e5a')
ax.set_ylabel('Power (mW)')
ax.set_xlabel('Time (s)')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')