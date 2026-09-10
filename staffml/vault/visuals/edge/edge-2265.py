import matplotlib.pyplot as plt
import numpy as np
import os

time = np.linspace(0, 10, 100)
# Arrival = 40 Hz, Service = 1000/25.5 = 39.2 Hz
queue_size = time * (40 - 39.21)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(time, queue_size, color='#c87b2a', linewidth=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Events in Queue')
ax.set_title('Unstable Sensor Fusion Queue Growth')
ax.grid(True, linestyle=':', alpha=0.6)

out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)