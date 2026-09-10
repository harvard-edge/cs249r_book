import os
import matplotlib.pyplot as plt
import numpy as np
t = np.linspace(0, 300, 600)
current = np.where(t % 100 < 2, 5, 0.01)
plt.plot(t, current, color='#3d9e5a')
plt.xlabel('Time (ms)')
plt.ylabel('Current (mA)')
plt.yscale('log')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')