import os
import matplotlib.pyplot as plt
import numpy as np
t = np.linspace(0, 0.2, 100)
energy = 0.009 - (0.06 * t)
fig, ax = plt.subplots(figsize=(5,3))
ax.plot(t, energy, color='#c87b2a', label='Capacitor Energy')
ax.axhline(0, color='red', linestyle='--', label='Depletion')
ax.set_ylabel('Energy (J)')
ax.set_xlabel('Time (s)')
ax.legend()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')