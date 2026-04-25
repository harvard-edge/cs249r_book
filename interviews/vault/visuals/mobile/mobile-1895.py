import os
import matplotlib.pyplot as plt
import numpy as np
t = np.linspace(0, 200, 400)
active = np.where(t % 50 < 5, 1, 0)
plt.plot(t, active, color='#3d9e5a', drawstyle='steps-pre')
plt.xlabel('Time (ms)')
plt.yticks([0, 1], ['Sleep', 'Active'])
plt.ylim(-0.2, 1.2)
out = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
plt.savefig(out, format='svg', bbox_inches='tight')