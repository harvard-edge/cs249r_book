import os
import matplotlib.pyplot as plt
import numpy as np
rho = np.linspace(0.1, 0.9, 100)
wq = (rho * 20) / (2 * (1 - rho))
plt.plot(rho, wq, color='#4a90c4')
plt.axhline(40, color='#c87b2a', linestyle='--')
plt.axvline(0.8, color='#3d9e5a', linestyle=':')
plt.xlabel('Utilization (rho)')
plt.ylabel('Avg Queue Wait Time (ms)')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')