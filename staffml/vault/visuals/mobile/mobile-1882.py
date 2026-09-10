import os
import numpy as np
import matplotlib.pyplot as plt

rho = np.linspace(0.1, 0.9, 100)
mm1 = rho / (1 - rho)
md1 = (rho**2) / (2 * (1 - rho))
plt.figure(figsize=(6,4))
plt.plot(rho, mm1, label='M/M/1', color='#4a90c4')
plt.plot(rho, md1, label='M/D/1', color='#3d9e5a')
plt.xlabel('Utilization')
plt.ylabel('Queue Length')
plt.legend()
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')