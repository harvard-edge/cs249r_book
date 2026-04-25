import os
import numpy as np
import matplotlib.pyplot as plt

rho = np.linspace(0.1, 0.95, 100)
queue_length = (rho**2) / (2 * (1 - rho))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(rho, queue_length, color='#c87b2a', linewidth=2)
ax.axvline(x=0.9, color='red', linestyle='--', label='Current Load (0.9)')
ax.set_xlabel('Utilization (ρ)')
ax.set_ylabel('Average Queue Length')
ax.set_title('Queueing Delay Hockey Stick Curve')
ax.legend()
plt.grid(True, linestyle='--', alpha=0.7)

out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)