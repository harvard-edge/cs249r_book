import os
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 10, 100)
ap_power = np.where((t > 4.8) & (t < 5.2), 2000, 0)
hub_power = np.full_like(t, 50)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(t, ap_power, label='App Processor', color='#c87b2a')
ax.plot(t, hub_power, label='Sensor Hub', color='#3d9e5a', lw=3)
ax.set_yscale('symlog')
ax.set_ylabel('Power (mW)')
ax.set_xlabel('Time')
ax.legend()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')