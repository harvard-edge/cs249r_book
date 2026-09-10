import os
import matplotlib.pyplot as plt
import numpy as np
lam = np.linspace(1, 6.5, 100)
service = 0.15
rho = lam * service
wq = (rho * service) / (2 * (1 - rho))
plt.plot(lam, wq, color='#4a90c4', linewidth=2)
plt.axvline(5, color='#c87b2a', linestyle='--')
plt.xlabel('Arrival Rate (Events/s)')
plt.ylabel('Queue Wait Time (s)')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')