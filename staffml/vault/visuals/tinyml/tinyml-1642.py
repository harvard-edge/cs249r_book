import os
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 100, 1000)
power = np.where(t < 5, 10, 0.01)

plt.figure(figsize=(6, 3))
plt.plot(t, power, color='#4a90c4', linewidth=2)
plt.fill_between(t, power, color='#cfe2f3')
plt.xlabel('Time (ms)')
plt.ylabel('Power (mW)')
plt.title('Duty Cycle Power Profile')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')