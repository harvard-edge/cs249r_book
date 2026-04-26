import os
import matplotlib.pyplot as plt
import numpy as np
t = np.linspace(0, 10, 200)
power = np.where(t % 1 < 0.1, 50, 2)
plt.plot(t, power, color='#c87b2a', drawstyle='steps-pre')
plt.xlabel('Time')
plt.ylabel('Power (mW)')
plt.ylim(0, 60)
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')