import matplotlib.pyplot as plt
import numpy as np
import os

rho = np.linspace(0.1, 0.95, 100)
q_depth = rho / (1 - rho)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(rho, q_depth, color='#4a90c4', linewidth=2.5)
ax.set_xlabel('Utilization (ρ)')
ax.set_ylabel('Avg Queue Depth')
ax.set_title('Queue Depth vs Utilization')
ax.grid(True, linestyle='--', alpha=0.6)

out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)