import os
import matplotlib.pyplot as plt
import numpy as np

rho = np.linspace(0.1, 0.9, 50)
wait_uniform = (rho) / (1 - rho) * 0.5
wait_bursty = (rho) / (1 - rho) * 2.5

plt.figure(figsize=(6, 3))
plt.plot(rho, wait_uniform, label='Uniform Arrivals', color='#3d9e5a')
plt.plot(rho, wait_bursty, label='Bursty Arrivals', color='#c87b2a', linestyle='--')
plt.xlabel('Utilization (rho)')
plt.ylabel('Expected Wait Time')
plt.legend()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')