import os
import numpy as np
import matplotlib.pyplot as plt

lambdas = np.linspace(10, 48, 50)
mu = 50
Wq = (lambdas / mu) / (mu - lambdas)

plt.figure(figsize=(6, 4))
plt.plot(lambdas, Wq * 1000, color='#c87b2a', lw=2)
plt.axvline(40, color='red', linestyle='--', label='Arrival=40')
plt.ylabel('Queue Wait Time (ms)')
plt.xlabel('Arrival Rate (chunks/s)')
plt.legend()

out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')