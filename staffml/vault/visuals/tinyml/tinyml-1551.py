import matplotlib.pyplot as plt
import numpy as np
import os

time = np.linspace(0, 10, 100)
queue = np.zeros_like(time)
queue[time <= 5] = time[time <= 5] * (1 - 1/1.2)
queue[time > 5] = np.maximum(0, queue[50] - (time[time > 5] - 5)*(1/1.2))

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(time, queue, color='#4a90c4', linewidth=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Backlogged Windows')
ax.set_title('Queue Depth During Vibration Burst')
ax.grid(True, linestyle=':', alpha=0.5)

out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)