import os
import matplotlib.pyplot as plt
import numpy as np
rho = np.linspace(0.1, 0.95, 100)
queue_len = rho**2 / (1 - rho) # M/D/1 approx queue length
plt.plot(rho, queue_len, color='#4a90c4')
plt.axvline(0.8, color='#3d9e5a', linestyle='--', label='Operating Point')
plt.xlabel('Utilization (rho)')
plt.ylabel('Mean Queue Length')
plt.legend()
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')