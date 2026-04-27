import os
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
rho = np.linspace(0.1, 0.9, 50)
latency = 1 / (40 - 40*rho)
ax.plot(rho, latency * 1000, color='red')
ax.set_xlabel('Utilization (rho)')
ax.set_ylabel('Latency (ms)')
ax.set_title('Queueing Latency vs Utilization')
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)