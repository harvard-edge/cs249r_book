import matplotlib.pyplot as plt
import numpy as np
import os

time = np.arange(0, 400)
progress = time % 60
failure = np.where(time == 300, 1, 0)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(time, progress, color='#3d9e5a', label='Progress')
ax.axvline(x=300, color='#c87b2a', linestyle='--', label='Power Loss')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Unsaved State (s)')
ax.set_title('Checkpointing RPO Timeline')
ax.legend()
ax.grid(True, linestyle=':', alpha=0.5)

out_path = os.environ.get('VISUAL_OUT_PATH', 'out.svg')
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close(fig)