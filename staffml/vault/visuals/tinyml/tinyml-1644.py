import os
import matplotlib.pyplot as plt
import numpy as np

arrival = np.linspace(10, 99, 100)
rho = arrival / 100.0
q_len = (rho**2) / (1 - rho)

plt.figure(figsize=(6, 3))
plt.plot(arrival, q_len, color='#4a90c4', lw=2)
plt.axvline(95, color='#c87b2a', linestyle='--', label='95 events/s')
plt.xlabel('Arrival Rate (events/sec)')
plt.ylabel('Expected Queue Length')
plt.legend()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')