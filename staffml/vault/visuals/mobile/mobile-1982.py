import os
import matplotlib.pyplot as plt
import numpy as np
t = np.array([0, 100, 200, 300])
q = np.array([15, 10, 5, 0])
fig, ax = plt.subplots(figsize=(5,3))
ax.plot(t, q, color='#4a90c4', linewidth=2, marker='o')
ax.set_ylabel('Queue Length')
ax.set_xlabel('Time (ms)')
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')