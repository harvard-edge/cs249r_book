import os
import matplotlib.pyplot as plt
import numpy as np

rho = np.linspace(0.1, 0.95, 50)
wait_mm1 = rho / (1 - rho)
wait_md1 = 0.5 * wait_mm1

plt.figure(figsize=(6, 3))
plt.plot(rho, wait_mm1, color='#c87b2a', label='M/M/1')
plt.plot(rho, wait_md1, color='#4a90c4', label='M/D/1')
plt.xlabel('Utilization (rho)')
plt.ylabel('Wait Time (normalized)')
plt.legend()
plt.grid(True)
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')