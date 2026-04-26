import os
import matplotlib.pyplot as plt
import numpy as np
intervals = np.linspace(0.5, 5, 50)
overhead = (intervals/2) + (24/intervals)*0.0833
fig, ax = plt.subplots(figsize=(5,3))
ax.plot(intervals, overhead, color='#c87b2a')
ax.axvline(2, color='red', linestyle='--', label='Optimal (2 hrs)')
ax.set_xlabel('Checkpoint Interval (hrs)')
ax.set_ylabel('Wasted Time')
ax.legend()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')