import os
import matplotlib.pyplot as plt
import numpy as np
t = np.linspace(0, 3, 300)
y = np.where(t % 1.0 < 0.02, 15, 0.05)
fig, ax = plt.subplots(figsize=(5,3))
ax.plot(t, y, color='#4a90c4')
ax.set_ylabel('Power (mW)')
ax.set_xlabel('Time (s)')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')