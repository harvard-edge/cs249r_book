import os
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(6, 3))
t = np.linspace(0, 33.2, 500)
power = np.where((t % 16.6) < 5.0, 2.0, 0.1)
ax.plot(t, power, color='#c87b2a')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Power (W)')
ax.set_title('60fps Duty Cycle')
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)