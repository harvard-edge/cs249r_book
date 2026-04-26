import os
import matplotlib.pyplot as plt
import numpy as np
t = np.linspace(0, 10, 1000)
y = np.where(t % 2 < 0.2, 10, 0.001)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(t, y, color='#4a90c4')
ax.set_yscale('log')
ax.set_ylabel('Power (mW)')
ax.set_xlabel('Time (ms)')
ax.set_title('Duty Cycling Power Profile')
plt.tight_layout()
plt.savefig(os.environ['VISUAL_OUT_PATH'], format='svg', bbox_inches='tight')