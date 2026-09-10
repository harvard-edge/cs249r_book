import matplotlib.pyplot as plt
import numpy as np
import os

time = np.linspace(0, 6, 100)
buffer = np.minimum(100, time * 20) # 20 packets net growth per second

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(time, buffer, color='#c87b2a', linewidth=2)
ax.axhline(y=100, color='red', linestyle='--', label='Buffer Capacity (100)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Queue Depth (Packets)')
ax.set_title('Queue Overflow Analysis')
ax.legend()
ax.grid(True, linestyle=':', alpha=0.5)

out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)