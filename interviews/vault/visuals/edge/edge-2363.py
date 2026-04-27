import os
import matplotlib.pyplot as plt
import numpy as np
rho = np.linspace(0.1, 0.95, 50)
latency = 1 / (50 * (1 - rho))
fig, ax = plt.subplots(figsize=(5,3))
ax.plot(rho, latency, color='#c87b2a')
ax.axvline(0.8, color='red', linestyle='--', label='80% Load (100ms)')
ax.set_xlabel('Utilization (rho)')
ax.set_ylabel('Response Time (s)')
ax.legend()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')