import os
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
rho = np.linspace(0.5, 0.98, 50)
lq = (rho**2) / (1 - rho)
ax.plot(rho, lq, color='#c87b2a', lw=2)
ax.axvline(0.95, color='r', linestyle='--', label='App State (rho=0.95)')
ax.set_ylabel('Queue Length')
ax.set_xlabel('Utilization')
ax.legend()
out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)