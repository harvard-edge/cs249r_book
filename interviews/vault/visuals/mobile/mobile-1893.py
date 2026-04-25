import os
import matplotlib.pyplot as plt
import numpy as np
lam = np.linspace(5, 38, 100)
mu = 40
rho = lam / mu
wq = rho / (mu - lam)
plt.plot(lam, wq, color='#4a90c4', linewidth=2)
plt.axvline(30, color='#c87b2a', linestyle='--')
plt.xlabel('Arrival Rate (FPS)')
plt.ylabel('Queuing Delay (s)')
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')