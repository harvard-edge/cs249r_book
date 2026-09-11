import matplotlib.pyplot as plt
import numpy as np
import os

time = np.linspace(0, 10, 1000)
power = np.full_like(time, 208) # Raised baseline in mW
power[(time > 2) & (time < 2.5)] = 2500
power[(time > 7) & (time < 7.5)] = 2500

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(time, power, color='#4a90c4')
ax.axhline(y=1, color='#3d9e5a', linestyle='--', label='Expected Sleep (1mW)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Power (mW)')
ax.set_title('Anomalous Duty Cycle Power Profile')
ax.legend()

out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)