import os
import matplotlib.pyplot as plt
import numpy as np

hours = np.arange(24)
traffic = 20 + 80 * np.exp(-((hours - 14)**2) / 4)
active_gpus = np.where((hours >= 13) & (hours <= 17), 100, 20)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(hours, traffic, label='Traffic Demand', color='#c87b2a', linewidth=2)
ax.step(hours, active_gpus, where='post', label='Active GPUs', color='#3d9e5a', linewidth=2)
ax.fill_between(hours, traffic, active_gpus, step='post', color='#d4edda', alpha=0.5, label='Idle Buffer')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('GPU Count')
ax.set_title('Duty Cycling Timeline')
ax.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.environ.get('VISUAL_OUT_PATH', 'out.svg'), format='svg', bbox_inches='tight')