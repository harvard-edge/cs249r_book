import os
import numpy as np
import matplotlib.pyplot as plt
rho = np.linspace(0, 0.95, 100)
delay = rho / (1 - rho)
plt.plot(rho, delay, color='#4a90c4', linewidth=2)
plt.xlabel('Utilization (rho)')
plt.ylabel('Queueing Delay')
plt.title('Delay Explosion')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')