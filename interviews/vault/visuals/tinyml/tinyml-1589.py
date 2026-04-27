import os
import numpy as np
import matplotlib.pyplot as plt
rho = np.linspace(0, 0.9, 50)
D = 10
delay = (rho * D) / (2 * (1 - rho))
plt.plot(rho, delay, color='#3d9e5a')
plt.xlabel('Utilization (rho)')
plt.ylabel('Delay (ms)')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')