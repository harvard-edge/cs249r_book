import os
import matplotlib.pyplot as plt
import numpy as np

rho = np.linspace(0.1, 0.95, 100)
L = rho / (1 - rho)
plt.figure(figsize=(6, 4))
plt.plot(rho, L, color='#c87b2a', linewidth=2)
plt.axvline(0.8, color='#4a90c4', linestyle='--', label='Current: rho=0.8')
plt.xlabel('Utilization (rho)')
plt.ylabel('Average Queue Length')
plt.legend()

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')