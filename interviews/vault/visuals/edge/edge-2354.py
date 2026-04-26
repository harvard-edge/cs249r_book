import os
import matplotlib.pyplot as plt
import numpy as np
rho = np.linspace(0, 0.95, 100)
wait = rho / (1 - rho)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(rho, wait, color='#c87b2a', linewidth=2)
ax.set_xlabel('Utilization (ρ)')
ax.set_ylabel('Queueing Delay')
ax.set_title('Hockey Stick Curve: Latency vs Utilization')
plt.grid(True)
plt.savefig(os.environ['VISUAL_OUT_PATH'], format='svg', bbox_inches='tight')