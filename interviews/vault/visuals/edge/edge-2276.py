import os
import matplotlib.pyplot as plt
import numpy as np
t = np.linspace(0, 2, 400)
power = np.where(t % 1.0 < 0.1, 2.5, 0.1)
plt.plot(t, power, color='#c87b2a', drawstyle='steps-pre')
plt.xlabel('Time (s)')
plt.ylabel('Power (W)')
plt.ylim(0, 3)
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')